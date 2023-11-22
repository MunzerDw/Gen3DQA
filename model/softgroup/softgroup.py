import torch
import time
import numpy as np
import torch.nn as nn
from evaluation.instance_segmentation import get_gt_instances, rle_encode
from evaluation.object_detection import get_gt_bbox
from common_ops.functions import softgroup_ops, common_ops
from loss import ClassificationLoss, MaskScoringLoss, IouScoringLoss
from evaluation.semantic_segmentation import *
from model.softgroup.module import TinyUnet
from model.softgroup.general_model import GeneralModel, clusters_voxelization, get_batch_offsets


class SoftGroup(GeneralModel):
    def __init__(self, model, data, optimizer, lr_decay, inference=None):
        super().__init__(model, data, optimizer, lr_decay, inference)
        output_channel = model.m
        self.instance_classes = data.scannet.classes - len(data.scannet.ignore_classes)

        """
            Top-down Refinement Block
        """
        self.tiny_unet = TinyUnet(output_channel)

        self.classification_branch = nn.Linear(output_channel, self.instance_classes + 1)

        self.mask_scoring_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, self.instance_classes + 1)
        )

        self.iou_score = nn.Linear(output_channel, self.instance_classes + 1)

    def forward(self, data_dict):
        output_dict = super().forward(data_dict)
        if data_dict["current_epoch"] > self.model.prepare_epochs:
            """
                Top-down Refinement Block
            """
            semantic_scores = output_dict["semantic_scores"].softmax(dim=-1)
            batch_idxs = data_dict["vert_batch_ids"]

            proposals_offset_list = []
            proposals_idx_list = []

            for class_id in range(self.data.scannet.classes):
                if class_id in self.data.scannet.ignore_classes:
                    continue
                scores = semantic_scores[:, class_id].contiguous()
                object_idxs = (scores > self.model.grouping_cfg.score_thr).nonzero().view(-1)
                if object_idxs.size(0) < self.model.test_cfg.min_npoint:
                    continue
                batch_idxs_ = batch_idxs[object_idxs].int()
                batch_offsets_ = get_batch_offsets(batch_idxs_, self.data.batch_size, output_dict["semantic_scores"].device)
                coords_ = data_dict["locs"][object_idxs]
                pt_offsets_ = output_dict["point_offsets"][object_idxs]
                idx, start_len = common_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
                                                              self.model.grouping_cfg.radius,
                                                              self.model.grouping_cfg.mean_active)

                proposals_idx, proposals_offset = softgroup_ops.sg_bfs_cluster(
                    self.model.grouping_cfg.class_numpoint_mean, idx.cpu(),
                    start_len.cpu(),
                    self.model.grouping_cfg.npoint_thr, class_id)
                proposals_idx[:, 1] = object_idxs.cpu()[proposals_idx[:, 1].long()].int()

                # merge proposals
                if len(proposals_offset_list) > 0:
                    proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                    proposals_offset += proposals_offset_list[-1][-1]
                    proposals_offset = proposals_offset[1:]
                if proposals_idx.size(0) > 0:
                    proposals_idx_list.append(proposals_idx)
                    proposals_offset_list.append(proposals_offset)
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)

            if proposals_offset.shape[0] > self.model.train_cfg.max_proposal_num:
                proposals_offset = proposals_offset[:self.model.train_cfg.max_proposal_num + 1]
                proposals_idx = proposals_idx[:proposals_offset[-1]]
                assert proposals_idx.shape[0] == proposals_offset[-1]

            proposals_offset = proposals_offset.cuda()
            output_dict["proposals_idx"] = proposals_idx
            output_dict["proposals_offset"] = proposals_offset

            inst_feats, inst_map = clusters_voxelization(
                clusters_idx=proposals_idx,
                clusters_offset=proposals_offset,
                feats=output_dict["point_features"],
                coords=data_dict["locs"],
                mode=4,
                device=output_dict["point_features"].device,
                **self.model.instance_voxel_cfg
            )

            inst_map = inst_map.long().cuda()

            feats = self.tiny_unet(inst_feats)

            # predict mask scores
            mask_scores = self.mask_scoring_branch(feats.features)
            output_dict["mask_scores"] = mask_scores[inst_map]
            output_dict["instance_batch_idxs"] = feats.coordinates[:, 0][inst_map]

            # predict instance cls and iou scores
            feats = self.global_pool(feats)
            output_dict["cls_scores"] = self.classification_branch(feats)
            output_dict["iou_scores"] = self.iou_score(feats)
            output_dict["object_proposals"] = feats

        return output_dict

    def global_pool(self, x, expand=False):
        indices = x.coordinates[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1,), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = softgroup_ops.global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool
        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x

    def _loss(self, data_dict, output_dict):
        losses, total_loss = super()._loss(data_dict, output_dict)

        total_loss += self.model.loss_weight[0] * losses["semantic_loss"] + \
                      self.model.loss_weight[1] * losses["offset_norm_loss"]

        if data_dict["current_epoch"] > self.model.prepare_epochs:
            proposals_idx = output_dict["proposals_idx"][:, 1].cuda()
            proposals_offset = output_dict["proposals_offset"]

            # calculate iou of clustered instance
            ious_on_cluster = common_ops.get_mask_iou_on_cluster(proposals_idx, proposals_offset,
                                                                 data_dict["instance_ids"],
                                                                 data_dict["instance_num_point"])

            # filter out background instances
            fg_inds = (data_dict["instance_semantic_cls"] != self.data.scannet.ignore_label)
            fg_instance_cls = data_dict["instance_semantic_cls"][fg_inds]
            fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

            # assign proposal to gt idx. -1: negative, 0 -> num_gts - 1: positive
            num_proposals = fg_ious_on_cluster.size(0)
            assigned_gt_inds = fg_ious_on_cluster.new_full((num_proposals,), -1, dtype=torch.long)

            # overlap > thr on fg instances are positive samples
            max_iou, argmax_iou = fg_ious_on_cluster.max(1)
            pos_inds = max_iou >= self.model.train_cfg.pos_iou_thr
            assigned_gt_inds[pos_inds] = argmax_iou[pos_inds]

            """classification loss"""
            # follow detection convention: 0 -> K - 1 are fg, K is bg
            labels = fg_instance_cls.new_full((num_proposals,), self.instance_classes)
            pos_inds = assigned_gt_inds >= 0
            labels[pos_inds] = fg_instance_cls[assigned_gt_inds[pos_inds]]
            classification_criterion = ClassificationLoss()
            labels = labels.long()
            classification_loss = classification_criterion(output_dict["cls_scores"], labels)
            losses["classification_loss"] = classification_loss

            """mask scoring loss"""
            mask_cls_label = labels[output_dict["instance_batch_idxs"].long()]
            slice_inds = torch.arange(0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
            mask_scores_sigmoid_slice = output_dict["mask_scores"].sigmoid()[slice_inds, mask_cls_label]

            mask_label, mask_label_mask = common_ops.get_mask_label(proposals_idx, proposals_offset,
                                                                    data_dict["instance_ids"],
                                                                    data_dict["instance_semantic_cls"],
                                                                    data_dict["instance_num_point"], ious_on_cluster,
                                                                    self.data.scannet.ignore_label,
                                                                    self.model.train_cfg.pos_iou_thr)

            mask_scoring_criterion = MaskScoringLoss(weight=mask_label_mask, reduction='sum')
            mask_scoring_loss = mask_scoring_criterion(mask_scores_sigmoid_slice, mask_label.float())
            mask_scoring_loss /= (torch.count_nonzero(mask_label_mask) + 1)
            losses["mask_scoring_loss"] = mask_scoring_loss
            """iou scoring loss"""
            ious = common_ops.get_mask_iou_on_pred(proposals_idx, proposals_offset, data_dict["instance_ids"],
                                                   data_dict["instance_num_point"],
                                                   mask_scores_sigmoid_slice.detach())
            fg_ious = ious[:, fg_inds]
            gt_ious, _ = fg_ious.max(1)
            slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
            iou_score_weight = labels < self.instance_classes
            iou_score_slice = output_dict["iou_scores"][slice_inds, labels]
            iou_scoring_criterion = IouScoringLoss(reduction="none")
            iou_scoring_loss = iou_scoring_criterion(iou_score_slice, gt_ious)
            iou_scoring_loss = iou_scoring_loss[iou_score_weight].sum() / (iou_score_weight.count_nonzero() + 1)
            losses["iou_scoring_loss"] = iou_scoring_loss
            total_loss += + self.model.loss_weight[3] * classification_loss + self.model.loss_weight[
                4] * mask_scoring_loss + self.model.loss_weight[5] * iou_scoring_loss

        """total loss"""
        return losses, total_loss

    def _get_pred_instances(self, scan_id, gt_xyz, proposals_idx, num_points, cls_scores, iou_scores, mask_scores,
                            num_ignored_classes):
        num_instances = cls_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []
        for i in range(self.instance_classes):
            cls_pred = cls_scores.new_full((num_instances,), i + 1, dtype=torch.long)
            cur_cls_scores = cls_scores[:, i]
            cur_iou_scores = iou_scores[:, i]
            cur_mask_scores = mask_scores[:, i]
            score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
            mask_pred = torch.zeros((num_instances, num_points), dtype=torch.bool, device="cpu")
            mask_inds = cur_mask_scores > self.model.test_cfg.mask_score_thr
            cur_proposals_idx = proposals_idx[mask_inds].long()
            mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = True

            # filter low score instance
            inds = cur_cls_scores > self.model.test_cfg.cls_score_thr
            cls_pred = cls_pred[inds]
            score_pred = score_pred[inds]
            mask_pred = mask_pred[inds]

            # filter too small instances
            npoint = torch.count_nonzero(mask_pred, dim=1)
            inds = npoint >= self.model.test_cfg.min_npoint
            cls_pred = cls_pred[inds]
            score_pred = score_pred[inds]
            mask_pred = mask_pred[inds]

            cls_pred_list.append(cls_pred)
            score_pred_list.append(score_pred)
            mask_pred_list.append(mask_pred)

        cls_pred = torch.cat(cls_pred_list).numpy()
        score_pred = torch.cat(score_pred_list).numpy()
        mask_pred = torch.cat(mask_pred_list).numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {'scan_id': scan_id, 'label_id': cls_pred[i], 'conf': score_pred[i],
                    'pred_mask': rle_encode(mask_pred[i])}
            pred_xyz = gt_xyz[mask_pred[i]]
            pred['pred_bbox'] = np.concatenate((pred_xyz.min(0), pred_xyz.max(0)))
            pred_instances.append(pred)
        return pred_instances
