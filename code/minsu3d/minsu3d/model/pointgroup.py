import torch
import torch.nn as nn
import time
import numpy as np
from minsu3d.evaluation.instance_segmentation import get_gt_instances, rle_encode
from minsu3d.evaluation.object_detection import get_gt_bbox
from minsu3d.common_ops.functions import pointgroup_ops, common_ops
from minsu3d.loss import ScoreLoss
from minsu3d.loss.utils import get_segmented_scores
from minsu3d.model.module import TinyUnet
from minsu3d.evaluation.semantic_segmentation import *
from minsu3d.model.general_model import GeneralModel, clusters_voxelization, get_batch_offsets


class PointGroup(GeneralModel):
    def __init__(self, model, data, optimizer, lr_decay, inference=None):
        super().__init__(model, data, optimizer, lr_decay, inference)
        output_channel = model.m

        """
            ScoreNet Block
        """
        self.score_net = TinyUnet(output_channel)
        self.score_branch = nn.Linear(output_channel, 1)

    def forward(self, data_dict):
        output_dict = super().forward(data_dict)
        if self.current_epoch > self.hparams.model.prepare_epochs or self.hparams.model.freeze_backbone:
            # get prooposal clusters
            batch_idxs = data_dict["vert_batch_ids"]
            semantic_preds = output_dict["semantic_scores"].max(1)[1]

            # set mask
            semantic_preds_mask = torch.ones_like(semantic_preds, dtype=torch.bool)
            for class_label in self.hparams.data.ignore_classes:
                semantic_preds_mask = semantic_preds_mask & (semantic_preds != class_label)
            object_idxs = torch.nonzero(semantic_preds_mask).view(-1)  # exclude predicted wall and floor

            batch_idxs_ = batch_idxs[object_idxs].int()
            batch_offsets_ = get_batch_offsets(batch_idxs_, self.hparams.data.batch_size, self.device)
            coords_ = data_dict["locs"][object_idxs]
            pt_offsets_ = output_dict["point_offsets"][object_idxs]

            semantic_preds_cpu = semantic_preds[object_idxs].cpu().int()
            object_idxs_cpu = object_idxs.cpu()

            idx_shift, start_len_shift = common_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_,
                                                                      batch_offsets_,
                                                                      self.hparams.model.cluster.cluster_radius,
                                                                      self.hparams.model.cluster.cluster_shift_meanActive)
            proposals_idx_shift, proposals_offset_shift = pointgroup_ops.pg_bfs_cluster(semantic_preds_cpu,
                                                                                        idx_shift.cpu(),
                                                                                        start_len_shift.cpu(),
                                                                                        self.hparams.model.cluster.cluster_npoint_thre)
            proposals_idx_shift[:, 1] = object_idxs_cpu[proposals_idx_shift[:, 1].long()].int()
            # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset_shift: (nProposal + 1), int
            # proposals_batchId_shift_all: (sumNPoint,) batch id

            idx, start_len = common_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_,
                                                          self.hparams.model.cluster.cluster_radius,
                                                          self.hparams.model.cluster.cluster_meanActive)
            proposals_idx, proposals_offset = pointgroup_ops.pg_bfs_cluster(semantic_preds_cpu, idx.cpu(),
                                                                            start_len.cpu(),
                                                                            self.hparams.model.cluster.cluster_npoint_thre)
            proposals_idx[:, 1] = object_idxs_cpu[proposals_idx[:, 1].long()].int()
            # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int

            proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
            proposals_offset_shift += proposals_offset[-1]
            proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
            proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))
            proposals_offset = proposals_offset.cuda()

            # proposals voxelization again
            proposals_voxel_feats, proposals_p2v_map = clusters_voxelization(
                clusters_idx=proposals_idx,
                clusters_offset=proposals_offset,
                feats=output_dict["point_features"],
                coords=data_dict["locs"],
                scale=self.hparams.model.score_scale,
                spatial_shape=self.hparams.model.score_fullscale,
                mode=4,
                device=self.device
            )
            # proposals_voxel_feats: (M, C) M: voxels
            # proposals_p2v_map: point2voxel map (sumNPoint,)
            # score
            score_feats = self.score_net(proposals_voxel_feats)
            pt_score_feats = score_feats.features[proposals_p2v_map.long().cuda()]  # (sumNPoint, C)
            proposals_score_feats = common_ops.roipool(pt_score_feats, proposals_offset)  # (nProposal, C)
            scores = self.score_branch(proposals_score_feats)  # (nProposal, 1)
            output_dict["proposal_scores"] = (scores, proposals_idx, proposals_offset)

        return output_dict

    def _loss(self, data_dict, output_dict):
        losses, total_loss = super()._loss(data_dict, output_dict)

        total_loss += self.hparams.model.loss_weight[0] * losses["semantic_loss"] + \
                      self.hparams.model.loss_weight[1] * losses["offset_norm_loss"] + \
                      self.hparams.model.loss_weight[2] * losses["offset_dir_loss"]

        if self.current_epoch > self.hparams.model.prepare_epochs:
            """score loss"""
            scores, proposals_idx, proposals_offset = output_dict["proposal_scores"]
            instance_pointnum = data_dict["instance_num_point"]
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int
            ious = common_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset, data_dict["instance_ids"],
                                      instance_pointnum)  # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, self.hparams.model.fg_thresh, self.hparams.model.bg_thresh)
            score_criterion = ScoreLoss()
            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            losses["score_loss"] = score_loss
            total_loss += self.hparams.model.loss_weight[3] * score_loss
        return losses, total_loss

    def validation_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self._feed(data_dict)
        losses, total_loss = self._loss(data_dict, output_dict)

        # log losses
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False,
                 on_epoch=True, sync_dist=True, batch_size=1)
        for key, value in losses.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)

        # log semantic prediction accuracy
        semantic_predictions = output_dict["semantic_scores"].max(1)[1].cpu().numpy()
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["sem_labels"].cpu().numpy(),
                                                       ignore_label=self.hparams.data.ignore_label)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["sem_labels"].cpu().numpy(),
                                                   ignore_label=self.hparams.data.ignore_label)
        self.log("val_eval/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=1)
        self.log("val_eval/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=1)

        if self.current_epoch > self.hparams.model.prepare_epochs:
            pred_instances = self._get_pred_instances(data_dict["scan_ids"][0],
                                                      data_dict["locs"].cpu().numpy(),
                                                      output_dict["proposal_scores"][0].cpu(),
                                                      output_dict["proposal_scores"][1].cpu(),
                                                      output_dict["proposal_scores"][2].size(0) - 1,
                                                      output_dict["semantic_scores"].cpu(),
                                                      len(self.hparams.data.ignore_classes))
            gt_instances = get_gt_instances(data_dict["sem_labels"].cpu(), data_dict["instance_ids"].cpu(),
                                            self.hparams.data.ignore_classes)
            gt_instances_bbox = get_gt_bbox(data_dict["locs"].cpu().numpy(),
                                            data_dict["instance_ids"].cpu().numpy(),
                                            data_dict["sem_labels"].cpu().numpy(), self.hparams.data.ignore_label,
                                            self.hparams.data.ignore_classes)

            return pred_instances, gt_instances, gt_instances_bbox

    def test_step(self, data_dict, idx):
        # prepare input and forward
        start_time = time.time()
        output_dict = self._feed(data_dict)
        end_time = time.time() - start_time

        sem_labels_cpu = data_dict["sem_labels"].cpu()
        semantic_predictions = output_dict["semantic_scores"].max(1)[1].cpu().numpy()
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions,
                                                       sem_labels_cpu.numpy(),
                                                       ignore_label=self.hparams.data.ignore_label)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, sem_labels_cpu.numpy(),
                                                   ignore_label=self.hparams.data.ignore_label)

        if self.current_epoch > self.hparams.model.prepare_epochs:
            pred_instances = self._get_pred_instances(data_dict["scan_ids"][0],
                                                      data_dict["locs"].cpu().numpy(),
                                                      output_dict["proposal_scores"][0].cpu(),
                                                      output_dict["proposal_scores"][1].cpu(),
                                                      output_dict["proposal_scores"][2].size(0) - 1,
                                                      output_dict["semantic_scores"].cpu(),
                                                      len(self.hparams.data.ignore_classes))
            gt_instances = get_gt_instances(sem_labels_cpu, data_dict["instance_ids"].cpu(),
                                            self.hparams.data.ignore_classes)
            gt_instances_bbox = get_gt_bbox(data_dict["locs"].cpu().numpy(),
                                            data_dict["instance_ids"].cpu().numpy(),
                                            data_dict["sem_labels"].cpu().numpy(), self.hparams.data.ignore_label,
                                            self.hparams.data.ignore_classes)
            return semantic_accuracy, semantic_mean_iou, pred_instances, gt_instances, gt_instances_bbox, end_time

    def _get_nms_instances(self, cross_ious, scores, threshold):
        """ non max suppression for 3D instance proposals based on cross ious and scores

        Args:
            ious (np.array): cross ious, (n, n)
            scores (np.array): scores for each proposal, (n,)
            threshold (float): iou threshold

        Returns:
            np.array: idx of picked instance proposals
        """
        ixs = np.argsort(-scores)  # descending order
        pick = []
        while len(ixs) > 0:
            i = ixs[0]
            pick.append(i)
            ious = cross_ious[i, ixs[1:]]
            remove_ixs = np.where(ious > threshold)[0] + 1
            ixs = np.delete(ixs, remove_ixs)
            ixs = np.delete(ixs, 0)

        return np.array(pick, dtype=np.int32)

    def _get_pred_instances(self, scan_id, gt_xyz, proposals_scores, proposals_idx, num_proposals, semantic_scores,
                            num_ignored_classes):
        semantic_pred_labels = semantic_scores.max(1)[1]
        proposals_score = torch.sigmoid(proposals_scores.view(-1))  # (nProposal,) float
        # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        # proposals_offset: (nProposal + 1), int, cpu

        N = semantic_scores.shape[0]

        proposals_mask = torch.zeros((num_proposals, N), dtype=torch.bool, device="cpu")  # (nProposal, N), int, cuda
        proposals_mask[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = True

        # score threshold & min_npoint mask
        proposals_npoint = torch.count_nonzero(proposals_mask, dim=1)
        proposals_thres_mask = torch.logical_and(proposals_score > self.hparams.model.test.TEST_SCORE_THRESH,
                                                 proposals_npoint > self.hparams.model.test.TEST_NPOINT_THRESH)

        proposals_score = proposals_score[proposals_thres_mask]
        proposals_mask = proposals_mask[proposals_thres_mask]

        # instance masks non_max_suppression
        if proposals_score.shape[0] == 0:
            pick_idxs = np.empty(0)
        else:
            proposals_mask_f = proposals_mask.float()  # (nProposal, N), float
            intersection = torch.mm(proposals_mask_f, proposals_mask_f.t())  # (nProposal, nProposal), float
            proposals_npoint = proposals_mask_f.sum(1)  # (nProposal), float, cuda
            proposals_np_repeat_h = proposals_npoint.unsqueeze(-1).repeat(1, proposals_npoint.shape[0])
            proposals_np_repeat_v = proposals_npoint.unsqueeze(0).repeat(proposals_npoint.shape[0], 1)
            cross_ious = intersection / (
                    proposals_np_repeat_h + proposals_np_repeat_v - intersection)  # (nProposal, nProposal), float, cuda
            pick_idxs = self._get_nms_instances(cross_ious.numpy(), proposals_score.numpy(),
                                                self.hparams.model.test.TEST_NMS_THRESH)  # int, (nCluster,)

        clusters_mask = proposals_mask[pick_idxs].numpy()  # int, (nCluster, N)
        score_pred = proposals_score[pick_idxs].numpy()  # float, (nCluster,)
        nclusters = clusters_mask.shape[0]
        instances = []
        for i in range(nclusters):
            cluster_i = clusters_mask[i]  # (N)
            pred = {'scan_id': scan_id, 'label_id': semantic_pred_labels[cluster_i][0].item() - num_ignored_classes + 1,
                    'conf': score_pred[i], 'pred_mask': rle_encode(cluster_i)}
            pred_inst = gt_xyz[cluster_i]
            pred['pred_bbox'] = np.concatenate((pred_inst.min(0), pred_inst.max(0)))
            instances.append(pred)
        return instances
