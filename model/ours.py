import json
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, ReduceLROnPlateau

sys.path.append("cider/pyciderevalcap/ciderD")
from cider.pyciderevalcap.ciderD.ciderD import CiderD
from loss.reward_criterion import RewardCriterion
from loss.softmax_ranking_loss import SoftmaxRankingLoss
from model.softgroup import SoftGroup
from model.vqa import VQA
from model.vqg import VQG
from optimizer import cosine_lr_decay

# from util.cider.cider import Cider
from util.utils import generate_mask, get_ious


class Ours(pl.LightningModule):
    def __init__(self, cfg, dataset=None, testing=False):
        super().__init__()
        self.save_hyperparameters()
        if dataset is not None:
            self.answer_vocabulary = dataset["answer_vocabulary"]
            self.question_vocabulary = dataset["question_vocabulary"]
            self.register_buffer(
                "answer_embeddings", torch.FloatTensor(dataset["answer_embeddings"])
            )
            self.register_buffer(
                "question_embeddings", torch.FloatTensor(dataset["question_embeddings"])
            )

        self.memory = {"train": {}, "val": {}}

        # backbone
        self.softgroup = SoftGroup(
            cfg.model.softgroup.model,
            cfg.data,
            cfg.model.optimizer,
            cfg.model.softgroup.lr_decay,
            None,
        )

        # map object proposals from SoftGroup to appropriate dimension
        self.op_map = nn.Sequential(nn.Linear(32, 300))

        # VQA
        self.vqa = VQA(cfg, self.answer_vocabulary, self.answer_embeddings)

        # VQG
        self.vqg = VQG(cfg, self.question_vocabulary, self.question_embeddings)
        if testing:
            self.vqa.decoder.testing = True
            self.vqg.decoder.testing = True

        # for logging samples randomly during training
        # will be updated every epoch for training and validation
        self.random_batch_idx = 0
        self.batch_i = 0
        self.turn_tf_off_every_n_epochs = cfg.model.trainer.check_val_every_n_epoch

        # for rl
        self.CiderD_scorer = Cider()
        self.Bleu_scorer = Bleu(4)

        # freeze modules
        self._freeze_modules()

    def forward(self, batch):
        # do SoftGroup forward pass if the object proposals are not precomputed (slower)
        if "object_proposals" not in batch:
            # backbone
            batch["current_epoch"] = self.current_epoch
            output_dict = self.softgroup._feed(batch)
            batch["softgroup_output"] = output_dict
            batch["object_proposals"] = output_dict["object_proposals"]

            # get predicted bboxes, their averages, and their predicted cls and iou scores
            (
                batch["pred_bboxes"],
                batch["pred_bboxes_avgs"],
                batch["pred_cls_scores"],
                batch["pred_iou_scores"],
            ) = self._get_predicted_bboxes(batch)

            # batch the object proposal to get per sample sequences with 0 paddings
            batch["object_proposals"] = self._batch_object_proposals(batch)

        # map object proposals to correct length
        object_proposals = batch["object_proposals"]
        object_proposals_mask = generate_mask(batch["object_proposals"])
        batch["object_proposals_mask"] = object_proposals_mask
        batch["object_proposals"] = self.op_map(object_proposals)

        # VQA
        if not self.hparams.cfg.model.freeze_vqa:
            if (
                self.hparams.cfg.model.activate_cider_loss
                and not self.vqa.decoder.testing
            ):
                self.vqa.decoder.baseline = True
                resume_training = self.training
                self.eval()
                with torch.no_grad():
                    # VQA baseline
                    batch_copy = self._copy_batch(batch)
                    self.vqa(batch_copy)
                    batch["seq_sample"] = batch_copy["seq_sample"].detach().clone()
                    batch["seq_logprobs_sample"] = (
                        batch_copy["seq_logprobs_sample"].detach().clone()
                    )
                    batch_copy.clear()
                    # VQG baseline
                    vqg_seq_sample = None
                    for i in range(batch["seq_sample"].shape[1]):
                        batch_copy = self._copy_batch(batch)
                        batch_copy[
                            "answer_embeddings"
                        ] = self.vqa.decoder.answer_embeddings[
                            batch["seq_sample"][:, i : i + 1, :].squeeze(1)
                        ].to(
                            object_proposals.device
                        )
                        batch_copy["answer_embeddings_mask"] = (
                            batch["seq_sample"][:, i : i + 1, :].squeeze(1) == 0
                        )
                        self.vqg(batch_copy)
                        if vqg_seq_sample is None:
                            vqg_seq_sample = (
                                batch_copy["seq_greedy"].unsqueeze(1).detach().clone()
                            )
                        else:
                            vqg_seq_sample = torch.cat(
                                [
                                    vqg_seq_sample,
                                    batch_copy["seq_greedy"]
                                    .unsqueeze(1)
                                    .detach()
                                    .clone(),
                                ],
                                dim=1,
                            )
                        batch_copy.clear()
                    batch["vqg_seq_sample"] = vqg_seq_sample
                if resume_training:
                    self.train()
                self.vqa.decoder.baseline = False

            # # for validation in eval mode without Dropout
            # if self.vqa.decoder.validation:
            #     resume_training = self.training
            #     self.eval()
            #     with torch.no_grad():
            #         batch_copy = batch.copy()
            #         for key in batch_copy:
            #             if torch.is_tensor(batch_copy[key]):
            #                 batch_copy[key] = batch_copy[key].detach().clone()
            #         self.vqa(batch_copy)
            #     batch["acc_seq_greedy"] = batch_copy["seq_greedy"].detach().clone()
            #     batch["acc_obj_conf"] = batch_copy["obj_conf"].detach().clone()
            #     if resume_training:
            #         self.train()
            #     batch_copy.clear()

            if (
                self.hparams.cfg.model.activate_cider_loss
                and not self.vqa.decoder.testing
            ):
                batch_copy = self._copy_batch(batch)
            self.vqa(batch)
            if (
                self.hparams.cfg.model.activate_cider_loss
                and not self.vqa.decoder.testing
            ):
                batch_copy["answer_embeddings"] = self.vqa.decoder.answer_embeddings[
                    batch["seq_greedy"]
                ].to(object_proposals.device)
                batch_copy["answer_embeddings_mask"] = batch["seq_greedy"] == 0
                self.vqg.eval()
                self.vqg(batch_copy)
                self.vqg.train()
                batch["vqg_seq_greedy"] = batch_copy["seq_greedy"].detach().clone()
                batch_copy.clear()

            if "gt_bbox" in batch:
                (
                    batch["target"],
                    batch["pred_bbox"],
                    batch["pred_cls"],
                ) = self._get_objloc_targets_and_predictions(batch)
            else:
                (
                    batch["pred_bbox"],
                    batch["pred_cls"],
                ) = self._get_objloc_predictions(batch)
        # VQG
        else:
            if (
                self.hparams.cfg.model.activate_cider_loss
                and not self.vqg.decoder.testing
            ):
                self.vqg.decoder.baseline = True
                resume_training = self.training
                self.eval()
                with torch.no_grad():
                    batch_copy = batch.copy()
                    for key in batch_copy:
                        if torch.is_tensor(batch_copy[key]):
                            batch_copy[key] = batch_copy[key].detach().clone()
                    self.vqg(batch_copy)
                batch["seq_sample"] = batch_copy["seq_sample"].detach().clone()
                batch["seq_logprobs_sample"] = (
                    batch_copy["seq_logprobs_sample"].detach().clone()
                )
                batch_copy.clear()
                if resume_training:
                    self.train()
                self.vqg.decoder.baseline = False

            self.vqg(batch)

        return batch

    def _copy_batch(self, batch):
        batch_copy = batch.copy()

        for key in batch_copy:
            if torch.is_tensor(batch_copy[key]):
                batch_copy[key] = batch_copy[key].detach().clone()

        return batch_copy

    def on_fit_start(self):
        tb = self.logger.experiment  # noqa

        layout = {
            "losses": {
                "loss": ["Multiline", ["loss/train", "loss/val"]],
                "vqa_loss": ["Multiline", ["vqa_loss/train", "vqa_loss/val"]],
                "vqg_loss": ["Multiline", ["vqg_loss/train", "vqg_loss/val"]],
                "vqa_objloc_loss": [
                    "Multiline",
                    ["vqa_objloc_loss/train", "vqa_objloc_loss/val"],
                ],
                "vqg_objloc_loss": [
                    "Multiline",
                    ["vqg_objloc_loss/train", "vqg_objloc_loss/val"],
                ],
                "vqg_cider_loss": [
                    "Multiline",
                    ["vqg_cider_loss/train", "vqg_cider_loss/val"],
                ],
                "vqg_reward": [
                    "Multiline",
                    ["vqg_reward/train", "vqg_reward/val"],
                ],
                "vqg_reward_baseline": [
                    "Multiline",
                    ["vqg_reward_baseline/train", "vqg_reward_baseline/val"],
                ],
                "vqg_reward_gen": [
                    "Multiline",
                    ["vqg_reward_gen/train", "vqg_reward_gen/val"],
                ],
                "vqa_cider_loss": [
                    "Multiline",
                    ["vqa_cider_loss/train", "vqa_cider_loss/val"],
                ],
                "vqa_reward": [
                    "Multiline",
                    ["vqa_reward/train", "vqa_reward/val"],
                ],
                "vqa_reward_baseline": [
                    "Multiline",
                    ["vqa_reward_baseline/train", "vqa_reward_baseline/val"],
                ],
                "vqa_reward_gen": [
                    "Multiline",
                    ["vqa_reward_gen/train", "vqa_reward_gen/val"],
                ],
            },
            "accuracies": {
                "Bleu_1": ["Multiline", ["Bleu_1/train", "Bleu_1/val"]],
                "Bleu_2": ["Multiline", ["Bleu_2/train", "Bleu_2/val"]],
                "Bleu_3": ["Multiline", ["Bleu_3/train", "Bleu_3/val"]],
                "Bleu_4": ["Multiline", ["Bleu_4/train", "Bleu_4/val"]],
                "CIDEr": ["Multiline", ["CIDEr/train", "CIDEr/val"]],
                "METEOR": ["Multiline", ["METEOR/train", "METEOR/val"]],
                "ROUGE_L": ["Multiline", ["ROUGE_L/train", "ROUGE_L/val"]],
                "word_accuracy": [
                    "Multiline",
                    ["word_accuracy/train", "word_accuracy/val"],
                ],
                "iou_avg": ["Multiline", ["iou_avg/train", "iou_avg/val"]],
                "iou_avg_25": ["Multiline", ["iou_avg_25/train", "iou_avg_25/val"]],
                "iou_avg_50": ["Multiline", ["iou_avg_50/train", "iou_avg_50/val"]],
            },
        }

        tb.add_custom_scalars(layout)

    def training_step(self, batch, batch_idx):
        # for logging text of a random batch every epoch
        self.batch_i += 1

        if self.hparams.cfg.model.precompute_softgroup_data:
            self._freeze_all_parameters()
            self._precompute_softgroup(batch)
            # dummy loss
            loss = nn.CrossEntropyLoss()
            input = torch.randn(3, 5, requires_grad=True)
            target = torch.empty(3, dtype=torch.long).random_(5)
            output = loss(input, target)
            return {"loss": output}

        # forward pass
        batch = self.forward(batch)

        # losses
        losses = {}

        if not self.hparams.cfg.model.freeze_softgroup:
            softgroup_losses, softgroup_total_loss = self.softgroup._loss(
                batch, batch["softgroup_output"]
            )
            losses["softgroup_total_loss"] = softgroup_total_loss

        if not self.hparams.cfg.model.freeze_vqa:
            if self.hparams.cfg.model.activate_cider_loss:
                vqa_cider_loss = self._cider_loss(batch, mode="vqa")
                vqa_objloc_loss = self._objloc_loss(batch, mode="vqa")
                losses["vqa_cider_loss"] = vqa_cider_loss
                losses["vqa_objloc_loss"] = vqa_objloc_loss
            else:
                vqa_loss = self._xe_loss(batch, mode="vqa")
                vqa_objloc_loss = self._objloc_loss(batch, mode="vqa")
                losses["vqa_loss"] = vqa_loss
                losses["vqa_objloc_loss"] = vqa_objloc_loss

        if not self.hparams.cfg.model.freeze_vqg:
            if self.hparams.cfg.model.activate_cider_loss:
                vqg_cider_loss = self._cider_loss(batch, mode="vqg")
                losses["vqg_cider_loss"] = vqg_cider_loss
            else:
                vqg_loss = self._xe_loss(batch, mode="vqg")
                losses["vqg_loss"] = vqg_loss

        loss_weights = {
            "softgroup_total_loss": 1.0,
            "vqa_loss": 1.0,
            "vqa_cider_loss": 1.0,
            "vqa_objloc_loss": 1.0,
            "vqg_loss": 1.0,
            "vqg_cider_loss": 1.0,
            "vqg_objloc_loss": 1.0,
        }

        total_loss = 0
        for loss, value in losses.items():
            total_loss += value * loss_weights[loss]
        losses["loss"] = total_loss

        # logging
        if "vqa_reward" in batch:
            self.log(
                f"vqa_reward/train",
                batch["vqa_reward"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
            self.log(
                f"vqa_reward_baseline/train",
                batch["vqa_reward_baseline"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
            self.log(
                f"vqa_reward_gen/train",
                batch["vqa_reward_gen"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
            self.log(
                f"vqg_reward_gen/train",
                batch["vqg_reward_gen"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
            self.log(
                f"vqg_reward_baseline/train",
                batch["vqg_reward_baseline"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
        if "vqg_reward" in batch:
            self.log(
                f"vqg_reward/train",
                batch["vqg_reward"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
            self.log(
                f"vqg_reward_baseline/train",
                batch["vqg_reward_baseline"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
            self.log(
                f"vqg_reward_gen/train",
                batch["vqg_reward_gen"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
        # loss
        for key, value in losses.items():
            self.log(
                f"{key}/train",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
        # remove other losses to free memory
        losses = {"loss": losses["loss"]}
        if ((self.current_epoch + 1) % self.turn_tf_off_every_n_epochs) == 0:
            # accuracies
            if not self.hparams.cfg.model.freeze_vqa:
                iou_accuracies = self._acc(batch, mode="vqa")
                for key, value in iou_accuracies.items():
                    self.log(
                        f"{key}/train",
                        value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=self.hparams.cfg.data.batch_size,
                    )
            elif not self.hparams.cfg.model.freeze_vqg:
                iou_accuracies = self._acc(batch, mode="vqg")
                for key, value in iou_accuracies.items():
                    self.log(
                        f"{key}/train",
                        value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=self.hparams.cfg.data.batch_size,
                    )
            # for scores - get prediction and ground truth texts
            pred_indices = batch["seq_greedy"]

            seqs = {"pred": {}, "gt": {}}

            for batch_idx, pred in enumerate(pred_indices):
                if not self.hparams.cfg.model.freeze_vqa:
                    gt_texts = batch["answers"][batch_idx]
                    gt_texts = [{"caption": seq} for seq in gt_texts]
                    idx2token = self.answer_vocabulary["idx2token"]
                else:
                    gt_texts = batch["question"][batch_idx]
                    gt_texts = [{"caption": gt_texts}]
                    idx2token = self.question_vocabulary["idx2token"]
                question_id = batch["question_id"][batch_idx]
                pred_text = [
                    {
                        "caption": " ".join(
                            [idx2token[token.item()] for token in pred[:-1]]
                        )
                        .replace(" <pad>", "")
                        .replace("<start> ", "")
                        .replace(" <end>", "")
                    }
                ]
                seqs["pred"][question_id] = pred_text
                seqs["gt"][question_id] = gt_texts

            losses["seqs"] = seqs
        # data
        self._log_samples(batch, batch_idx, "train")

        return losses

    def on_train_epoch_start(self):
        self.random_batch_idx = random.randint(0, self.trainer.num_training_batches - 1)
        self.batch_i = 0
        if ((self.current_epoch + 1) % self.turn_tf_off_every_n_epochs) == 0:
            self.vqa.decoder.validation = True
            self.vqg.decoder.validation = True
        else:
            self.vqa.decoder.validation = False
            self.vqg.decoder.validation = False

    def training_epoch_end(self, training_step_outputs):
        if self.hparams.cfg.model.precompute_softgroup_data:
            self._save_precomputed_softgroup_data("train")
            return

        if ((self.current_epoch + 1) % self.turn_tf_off_every_n_epochs) == 0:
            # combine step outputs
            tokenizer = PTBTokenizer()
            pred_seqs = {"pred": {}, "gt": {}}
            outputs = [step_output["seqs"] for step_output in training_step_outputs]
            for output in outputs:
                pred_seqs["pred"].update(output["pred"])
                pred_seqs["gt"].update(output["gt"])
            pred_seqs["pred"] = tokenizer.tokenize(pred_seqs["pred"])
            pred_seqs["gt"] = tokenizer.tokenize(pred_seqs["gt"])

            # calculate scores
            scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(), "METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
            ]
            scores_results = {}
            for scorer, method in scorers:
                score, scores = scorer.compute_score(pred_seqs["gt"], pred_seqs["pred"])
                if type(method) == list:
                    for sc, _, m in zip(score, scores, method):
                        scores_results[m] = sc * 100
                else:
                    scores_results[method] = score * 100

            for key, value in scores_results.items():
                self.logger.experiment.add_scalar(
                    f"{key}/train", value, self.global_step - 1
                )

        # update learning rate
        # if not self.hparams.cfg.model.activate_cider_loss:
        cosine_lr_decay(
            self.trainer.optimizers[0],
            self.hparams.cfg.model.optimizer.lr,
            self.current_epoch,
            self.hparams.cfg.model.lr_decay.decay_start_epoch,
            self.hparams.cfg.model.lr_decay.decay_stop_epoch,
            1e-6,
        )

    def validation_step(self, batch, batch_idx):
        # for logging text of a random batch every epoch
        self.batch_i += 1

        if self.hparams.cfg.model.precompute_softgroup_data:
            self._freeze_all_parameters()
            self._precompute_softgroup(batch, mode="val")
            # dummy loss
            loss = nn.CrossEntropyLoss()
            input = torch.randn(3, 5, requires_grad=True)
            target = torch.empty(3, dtype=torch.long).random_(5)
            output = loss(input, target)
            return {"loss": output}

        # forward pass
        batch = self.forward(batch)

        # losses
        losses = {}

        if not self.hparams.cfg.model.freeze_softgroup:
            softgroup_losses, softgroup_total_loss = self.softgroup._loss(
                batch, batch["softgroup_output"]
            )
            losses["softgroup_total_loss"] = softgroup_total_loss

        if not self.hparams.cfg.model.freeze_vqa:
            if self.hparams.cfg.model.activate_cider_loss:
                vqa_cider_loss = self._cider_loss(batch, mode="vqa")
                vqa_objloc_loss = self._objloc_loss(batch, mode="vqa")
                losses["vqa_cider_loss"] = vqa_cider_loss
                losses["vqa_objloc_loss"] = vqa_objloc_loss
            else:
                vqa_loss = self._xe_loss(batch, mode="vqa")
                vqa_objloc_loss = self._objloc_loss(batch, mode="vqa")
                losses["vqa_loss"] = vqa_loss
                losses["vqa_objloc_loss"] = vqa_objloc_loss

        if not self.hparams.cfg.model.freeze_vqg:
            if self.hparams.cfg.model.activate_cider_loss:
                vqg_cider_loss = self._cider_loss(batch, mode="vqg")
                losses["vqg_cider_loss"] = vqg_cider_loss
            else:
                vqg_loss = self._xe_loss(batch, mode="vqg")
                losses["vqg_loss"] = vqg_loss

        loss_weights = {
            "softgroup_total_loss": 1.0,
            "vqa_loss": 1.0,
            "vqa_cider_loss": 1.0,
            "vqa_objloc_loss": 1.0,
            "vqg_loss": 1.0,
            "vqg_cider_loss": 1.0,
            "vqg_objloc_loss": 1.0,
        }

        total_loss = 0
        for loss, value in losses.items():
            total_loss += value * loss_weights[loss]
        losses["loss"] = total_loss

        # logging
        if "vqa_reward" in batch:
            self.log(
                f"vqa_reward/val",
                batch["vqa_reward"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
            self.log(
                f"vqa_reward_baseline/val",
                batch["vqa_reward_baseline"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
            self.log(
                f"vqa_reward_gen/val",
                batch["vqa_reward_gen"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
            self.log(
                f"vqg_reward_gen/val",
                batch["vqg_reward_gen"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
            self.log(
                f"vqg_reward_baseline/val",
                batch["vqg_reward_baseline"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
        if "vqg_reward" in batch:
            self.log(
                f"vqg_reward/val",
                batch["vqg_reward"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
            self.log(
                f"vqg_reward_baseline/val",
                batch["vqg_reward_baseline"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
            self.log(
                f"vqg_reward_gen/val",
                batch["vqg_reward_gen"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
        # loss
        for key, value in losses.items():
            self.log(
                f"{key}/val",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.hparams.cfg.data.batch_size,
            )
        # remove other losses to free memory
        losses = {"loss": losses["loss"]}
        if ((self.current_epoch + 1) % self.turn_tf_off_every_n_epochs) == 0:
            # accuracies
            if not self.hparams.cfg.model.freeze_vqa:
                iou_accuracies = self._acc(batch, mode="vqa")
                for key, value in iou_accuracies.items():
                    self.log(
                        f"{key}/val",
                        value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=self.hparams.cfg.data.batch_size,
                    )
            elif not self.hparams.cfg.model.freeze_vqg:
                iou_accuracies = self._acc(batch, mode="vqg")
                for key, value in iou_accuracies.items():
                    self.log(
                        f"{key}/val",
                        value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=self.hparams.cfg.data.batch_size,
                    )
            # for scores - get prediction and ground truth texts
            pred_indices = batch["seq_greedy"]

            seqs = {"pred": {}, "gt": {}}

            for batch_idx, pred in enumerate(pred_indices):
                if not self.hparams.cfg.model.freeze_vqa:
                    gt_texts = batch["answers"][batch_idx]
                    gt_texts = [{"caption": answer} for answer in gt_texts]
                    idx2token = self.answer_vocabulary["idx2token"]
                else:
                    gt_texts = batch["question"][batch_idx]
                    gt_texts = [{"caption": gt_texts}]
                    idx2token = self.question_vocabulary["idx2token"]
                question_id = batch["question_id"][batch_idx]
                pred_text = [
                    {
                        "caption": " ".join(
                            [idx2token[token.item()] for token in pred[:-1]]
                        )
                        .replace(" <pad>", "")
                        .replace("<start> ", "")
                        .replace(" <end>", "")
                    }
                ]
                seqs["pred"][question_id] = pred_text
                seqs["gt"][question_id] = gt_texts

            losses["seqs"] = seqs
        # data
        self._log_samples(batch, batch_idx, "val")

        return losses

    def on_validation_epoch_start(self):
        self.random_batch_idx = random.randint(0, self.trainer.num_val_batches[0] - 1)
        self.batch_i = 0
        if ((self.current_epoch + 1) % self.turn_tf_off_every_n_epochs) == 0:
            self.vqa.decoder.validation = True
            self.vqg.decoder.validation = True
        else:
            self.vqa.decoder.validation = False
            self.vqg.decoder.validation = False

    def validation_epoch_end(self, validation_step_outputs):
        if self.hparams.cfg.model.precompute_softgroup_data:
            self._save_precomputed_softgroup_data("val")
            return

        if ((self.current_epoch + 1) % self.turn_tf_off_every_n_epochs) == 0:
            # combine step outputs
            tokenizer = PTBTokenizer()
            pred_seqs = {"pred": {}, "gt": {}}
            outputs = [step_output["seqs"] for step_output in validation_step_outputs]
            for output in outputs:
                pred_seqs["pred"].update(output["pred"])
                pred_seqs["gt"].update(output["gt"])
            pred_seqs["pred"] = tokenizer.tokenize(pred_seqs["pred"])
            pred_seqs["gt"] = tokenizer.tokenize(pred_seqs["gt"])

            # calculate scores
            scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(), "METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
            ]
            scores_results = {}
            for scorer, method in scorers:
                score, scores = scorer.compute_score(pred_seqs["gt"], pred_seqs["pred"])
                if type(method) == list:
                    for sc, _, m in zip(score, scores, method):
                        scores_results[m] = sc * 100
                else:
                    scores_results[method] = score * 100

            for key, value in scores_results.items():
                self.log(f"{key}/val", value)

    def test_step(self, batch, batch_idx):
        # forward pass
        batch = self.forward(batch)

        # outputs
        # answer & question
        idx2token = self.answer_vocabulary["idx2token"]

        pred_indices = batch["vqa_output"]

        pred_anwers = [
            [idx2token[token.item()] for token in answer] for answer in pred_indices
        ]
        questions = batch["question"]

        question = (
            questions[0]
            .replace(" <pad>", "")
            .replace("<start> ", "")
            .replace(" <end>", "")
        )
        prediction = [
            " ".join(pred_anwer)
            .replace("<start> ", "")
            .replace(" <end>", "")
            .replace(" <pad>", "")
            for pred_anwer in pred_anwers
        ]
        gt_answer = batch["answers"][0]
        scene_id = batch["scan_ids"][0]
        question_id = batch["question_id"][0]

        # pred bbox
        pred_bbox = batch["pred_bbox"][0].cpu().tolist()

        # pred bbox class
        class_names = self.hparams.cfg.data.scannet.class_names
        pred_bbox_cls = batch["pred_cls"][0]
        pred_object_class = (
            class_names[pred_bbox_cls] if pred_bbox_cls < 18 else "background"
        )

        # gt bbox
        gt_bbox = batch["gt_bbox"][0].cpu().tolist()
        # gt_bboxes = np.array(batch["gt_bboxes"][0]).tolist()

        # iou
        iou = get_ious(
            np.expand_dims(np.array(gt_bbox), axis=0),
            np.expand_dims(np.array(pred_bbox), axis=0),
        )[0]
        # pred_bbox_repeated = np.expand_dims(np.array(pred_bbox), axis=0).repeat(len(gt_bboxes), axis=0)
        # ious = get_ious(
        #     np.array(gt_bboxes),
        #     pred_bbox_repeated
        # )
        # iou = max(ious)

        # result
        result = {
            "scene_id": scene_id,
            "question_id": question_id,
            "question": question,
            "gt_answers": gt_answer,
            "answer_top10": prediction,
            "gt_object_classes": batch["object_names"][0],
            "pred_object_class": pred_object_class,
            "gt_bbox": gt_bbox,
            # "gt_bbox": gt_bboxes,
            "bbox": pred_bbox,
            "iou": iou,
        }

        return result

    def test_epoch_end(self, test_step_outputs):
        # save results
        output_path = os.path.join(
            self.hparams.cfg.exp_output_root_path,
            self.hparams.cfg.model.experiment_name,
            "inference",
        )
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, "predictions.json")
        json.dump(test_step_outputs, open(output_path, "w"), indent=4)
        print(f"saved ({len(test_step_outputs)}) predictions to: {output_path}")

    def predict_step(self, batch, batch_idx):
        # forward pass
        batch = self.forward(batch)

        # outputs
        # answer & question
        idx2token = self.answer_vocabulary["idx2token"]

        pred_indices = batch["vqa_output"]

        pred_anwers = [
            [idx2token[token.item()] for token in answer] for answer in pred_indices
        ]
        questions = batch["question"]

        question = (
            questions[0]
            .replace(" <pad>", "")
            .replace("<start> ", "")
            .replace(" <end>", "")
        )
        prediction = (
            " ".join(pred_anwers[0])
            .replace("<start> ", "")
            .replace(" <end>", "")
            .replace(" <pad>", "")
        )
        scene_id = batch["scan_ids"][0]
        question_id = batch["question_id"][0]

        # pred bbox
        pred_bbox = batch["pred_bbox"][0].cpu().tolist()

        # pred bbox class
        class_names = self.hparams.cfg.data.scannet.class_names
        pred_bbox_cls = batch["pred_cls"][0]
        pred_object_class = (
            class_names[pred_bbox_cls] if pred_bbox_cls < 18 else "background"
        )

        # result
        result = {
            "scene_id": scene_id,
            "question_id": question_id,
            "question": question,
            "answer_top10": [prediction],
            "pred_object_class": pred_object_class,
            "bbox": pred_bbox,
        }

        return result

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        #     self.parameters(),
        #     lr=self.hparams.cfg.model.optimizer.lr,
        #     betas=(0.9, 0.999),
        #     eps=1e-08,
        #     weight_decay=0.0001,
        #     amsgrad=False,
        # )

        # def lr_scaler(epoch):
        #     lr_scale = 1.0
        #     if epoch <= self.hparams.cfg.model.optimizer.warm_up_epochs:
        #         # warm up lr
        #         lr_scale = 0.9 ** (
        #             self.hparams.cfg.model.optimizer.warm_up_epochs - epoch
        #         )
        #     elif epoch >= 90:
        #         lr_scale = 0.995**epoch

        #     return lr_scale

        # scheduler = LambdaLR(optimizer, lr_lambda=lr_scaler)

        # return [optimizer], [scheduler]

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.cfg.model.optimizer.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0001,
            amsgrad=False,
        )

        return optimizer

    # losses

    def _reward(self, res, gts, opt, mode="vqa"):
        CiderD_scorer = self.CiderD_scorer
        Bleu_scorer = self.Bleu_scorer
        device = res.device

        res = res.detach().cpu().numpy()

        # stringify sentences
        res_strings = [
            self._array_to_str(res_i, mode=mode)
            for res_group in res
            for res_i in res_group
        ]
        gts_strings = [
            [self._array_to_str(gts_i, mode=mode) for gts_i in gts_group]
            for gts_group in gts
        ]
        gts_strings = [
            gts_str_group for gts_str_group in gts_strings for _ in range(res.shape[1])
        ]

        # format sentences for scorers
        # for CIDEr-D
        res_ = [
            {"image_id": i, "caption": [res_strings[i]]}
            for i in range(len(res_strings))
        ]
        res__ = {i: [res_strings[i]] for i in range(len(res_))}
        gts_ = {i: gts_strings[i] for i in range(len(gts_strings))}

        # calculate scores
        if opt["cider_reward_weight"] > 0:
            _, cider_scores = CiderD_scorer.compute_score(gts_, res__)
        else:
            cider_scores = 0
        if opt["bleu_reward_weight"] > 0:
            _, bleu_scores = Bleu_scorer.compute_score(gts_, res__)
            bleu_scores = np.array(bleu_scores[3])
        else:
            bleu_scores = 0

        # combine scores
        scores = (
            opt["cider_reward_weight"] * cider_scores
            + opt["bleu_reward_weight"] * bleu_scores
        )

        # reshape
        batch_beam_scores = scores.reshape(res.shape[:2])  # batch_size, beam_size
        batch_scores = batch_beam_scores.mean(axis=1)  # batch_size

        return torch.from_numpy(batch_scores).to(device)

    def _cider_loss(self, batch, mode="vqa"):
        # predictions
        gen = batch["seq_greedy"].unsqueeze(1)  # batch_size, beam_size, MAX_LEN
        baseline = batch["seq_sample"]  # batch_size, beam_size, MAX_LEN
        logprobs = batch["seq_logprobs_greedy"].unsqueeze(
            1
        )  # batch_size, beam_size, MAX_LEN, vocab_size
        vqg_gen = batch["vqg_seq_greedy"].unsqueeze(1)  # batch_size, beam_size, MAX_LEN
        vqg_baseline = batch["vqg_seq_sample"]  # batch_size, beam_size, MAX_LEN

        # ground truth
        indicies = "answer_indicies" if mode == "vqa" else "question_indicies"
        data_gts = batch[indicies][:, 1:].unsqueeze(1).tolist()
        vqg_data_gts = batch["question_indicies"][:, 1:].unsqueeze(1).tolist()

        # loss
        # rewards
        baseline_reward = self._reward(
            baseline,
            data_gts,
            {"cider_reward_weight": 1.0, "bleu_reward_weight": 0.0},
            mode=mode,
        )
        gen_reward = self._reward(
            gen,
            data_gts,
            {"cider_reward_weight": 1.0, "bleu_reward_weight": 0.0},
            mode=mode,
        )
        vqg_baseline_reward = self._reward(
            vqg_baseline,
            vqg_data_gts,
            {"cider_reward_weight": 1.0, "bleu_reward_weight": 0.0},
            mode="vqg",
        )
        vqg_gen_reward = self._reward(
            vqg_gen,
            vqg_data_gts,
            {"cider_reward_weight": 1.0, "bleu_reward_weight": 0.0},
            mode="vqg",
        )
        reward_diff = (
            gen_reward - baseline_reward
        ) + self.hparams.cfg.model.vqg.factor * (vqg_gen_reward - vqg_baseline_reward)

        # policy gradient
        criterion = RewardCriterion()
        loss = criterion(logprobs, gen.detach(), reward_diff)

        batch[mode + "_reward"] = (gen_reward - baseline_reward).mean()
        batch[mode + "_reward_baseline"] = baseline_reward.mean()
        batch[mode + "_reward_gen"] = gen_reward.mean()
        batch["vqg" + "_reward_gen"] = vqg_gen_reward.mean()
        batch["vqg" + "_reward_baseline"] = vqg_baseline_reward.mean()
        batch["vqg" + "_reward"] = (vqg_gen_reward - vqg_baseline_reward).mean()

        return loss

    def _objloc_loss(self, batch, mode):
        # prediction
        obj_conf = batch["obj_conf"].cpu()  # b, num of object_proposals

        # ground truth
        target = batch["target"]

        # loss
        loss_fn = SoftmaxRankingLoss()
        # loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(obj_conf, target)

        return loss

    def _xe_loss(self, batch, mode):
        # prediction
        output = "vqg_output" if mode == "vqg" else "vqa_output"
        output = batch[output]

        # ground truth
        indicies = "question_indicies" if mode == "vqg" else "answer_indicies"
        question_indicies = batch[indicies]
        question_indicies = question_indicies[:, 1:]

        # loss
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.question_vocabulary["token2idx"]["<pad>"]
        )
        loss = loss_fn(
            output.reshape(-1, output.shape[2]),
            question_indicies.reshape(-1),
        )

        return loss

    # accuracies

    def _acc(self, batch, mode="vqa"):
        accuracies = {}

        # iou
        if mode == "vqa":
            pred_bbox = batch["pred_bbox"].cpu().numpy()
            gt_bbox = batch["gt_bbox"].cpu().numpy()

            ious = get_ious(pred_bbox, gt_bbox)
            batch["pred_bbox_ious"] = ious
            iou_avg = sum(ious) / len(ious)
            iou_avg_25 = len([x for x in ious if x >= 0.25]) / len(ious)
            iou_avg_50 = len([x for x in ious if x >= 0.5]) / len(ious)
            iou_accuracies = {
                "iou_avg": iou_avg,
                "iou_avg_25": iou_avg_25,
                "iou_avg_50": iou_avg_50,
            }
            accuracies.update(iou_accuracies)

        # word per word accuracy
        indices = "question_indicies" if mode == "vqg" else "answer_indicies"
        embeddings_mask = (
            "question_embeddings_mask" if mode == "vqg" else "answer_embeddings_mask"
        )
        gt_indices = batch[indices][:, 1:]
        pred_indices = batch["seq_greedy"]
        embeddings_mask = batch[embeddings_mask]

        correct = 0
        total = 0
        for batch_idx, pred in enumerate(pred_indices):
            mask = ~embeddings_mask[batch_idx]
            pred = pred[: mask.shape[0]][mask]
            gt = gt_indices[batch_idx][mask]
            same = pred == gt
            correct += same.sum().item()
            total += pred.shape[0]
        word_accuracy = {"word_accuracy": correct / total}
        accuracies.update(word_accuracy)

        return accuracies

    # logging

    def _log_samples(self, batch, batch_idx, split):
        try:
            if self.batch_i == self.random_batch_idx + 1:
                if not self.hparams.cfg.model.freeze_vqa:
                    idx2token = self.answer_vocabulary["idx2token"]
                    gt_indices = batch["answer_indicies"]
                    pred_indices = batch["vqa_output"]
                else:
                    idx2token = self.question_vocabulary["idx2token"]
                    gt_indices = batch["question_indicies"]
                    pred_indices = batch["vqg_output"]
                if ((self.current_epoch + 1) % self.turn_tf_off_every_n_epochs) == 0:
                    pred_indices = batch["seq_greedy"]

                gt_seqs = [
                    [idx2token[token.item()] for token in seq] for seq in gt_indices
                ]
                pred_seqs = [
                    [idx2token[token.item()] for token in seq] for seq in pred_indices
                ]
                questions = batch["question"]
                answers = batch["answers"]

                rand_idx = random.randrange(len(questions))
                question = (
                    questions[rand_idx]
                    .replace(" <pad>", "")
                    .replace("<", "(")
                    .replace(">", ")")
                )
                answer = (
                    answers[rand_idx][0]
                    .replace(" <pad>", "")
                    .replace("<", "(")
                    .replace(">", ")")
                )
                prediction = (
                    " ".join(pred_seqs[rand_idx])
                    .replace(" <pad>", "")
                    .replace("<", "(")
                    .replace(">", ")")
                )
                gt_truth = (
                    " ".join(gt_seqs[rand_idx])
                    .replace(" <pad>", "")
                    .replace("<", "(")
                    .replace(">", ")")
                )
                scene_id = batch["scan_ids"][rand_idx]
                question_id = batch["question_id"][rand_idx]
                # iou = batch["pred_bbox_ious"][rand_idx]
                iou = 0.0

                self.logger.experiment.add_text(
                    f"vqa/{split}",
                    "QUESTION: " + question + "  \n"
                    "ANSWER: "
                    + answer
                    + "  \n"
                    + "PREDICTION: "
                    + prediction
                    + "  \n"
                    + "GT: "
                    + gt_truth
                    + "  \n"
                    + "SCENE_ID: "
                    + scene_id
                    + "  \n"
                    + "QUESTION_ID: "
                    + question_id
                    + "  \n"
                    + "PRED_BBOX_IOU: "
                    + str(iou),
                    self.current_epoch,
                )
        except:
            pass

    # custom functions

    def _array_to_str(self, arr, mode="vqa"):
        if mode == "vqa":
            token2idx = self.vqa.decoder.answer_vocabulary["token2idx"]
            idx2token = self.vqa.decoder.answer_vocabulary["idx2token"]
        else:
            token2idx = self.vqg.decoder.question_vocabulary["token2idx"]
            idx2token = self.vqg.decoder.question_vocabulary["idx2token"]
        out = ""
        for i in range(len(arr)):
            out += str(arr[i]) + " "
            # out += idx2token[arr[i]] + " "
            if arr[i] == token2idx["<end>"]:
                break
        return out.strip()

    def _save_precomputed_softgroup_data(self, split):
        output_path = os.path.join(
            self.hparams.cfg.data.scannet.precompute_output_path,
            f"{split}_scene_info.json",
        )
        json.dump(self.memory[split], open(output_path, "w"), sort_keys=True, indent=4)
        print(f"saved {split} scenes to: {output_path}")

    def _precompute_softgroup(self, batch, mode="train"):
        # backbone
        batch["current_epoch"] = self.current_epoch
        output_dict = self.softgroup._feed(batch)
        batch["softgroup_output"] = output_dict
        batch["object_proposals"] = output_dict["object_proposals"]

        # get predicted bboxes, their averages, and their predicted cls and iou scores
        (
            batch["pred_bboxes"],
            batch["pred_bboxes_avgs"],
            batch["pred_cls_scores"],
            batch["pred_iou_scores"],
        ) = self._get_predicted_bboxes(batch)

        # batch the object proposal to get per sample sequences with 0 paddings
        batch["object_proposals"] = self._batch_object_proposals(batch)
        object_proposals_mask = generate_mask(batch["object_proposals"])
        batch["object_proposals_mask"] = object_proposals_mask

        for batch_idx, scene_id in enumerate(batch["scan_ids"]):
            batch_idx_mask = ~batch["object_proposals_mask"][batch_idx]
            pred_bboxes = batch["pred_bboxes"][batch_idx][batch_idx_mask]
            pred_bboxes_avgs = batch["pred_bboxes_avgs"][batch_idx][batch_idx_mask]
            pred_cls_scores = batch["pred_cls_scores"][batch_idx]
            pred_iou_scores = batch["pred_iou_scores"][batch_idx]
            object_proposals = batch["object_proposals"][batch_idx][batch_idx_mask]

            self.memory[mode][scene_id] = {
                "pred_bboxes": pred_bboxes.tolist(),
                "pred_bboxes_avgs": pred_bboxes_avgs.tolist(),
                "pred_cls_scores": pred_cls_scores,
                "pred_iou_scores": pred_iou_scores,
                "object_proposals": object_proposals.tolist(),
            }

            print(
                f"got data for scene {scene_id} ({mode}) - {object_proposals.shape[0]} object proposals, {pred_bboxes.shape[0]} predicted bboxes, {len(pred_cls_scores)} cls scores"
            )

        return batch

    def _freeze_all_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def _freeze_modules(self):
        if self.hparams.cfg.model.freeze_softgroup:
            for param in self.softgroup.parameters():
                param.requires_grad = False

        if self.hparams.cfg.model.freeze_vqa:
            for param in self.vqa.parameters():
                param.requires_grad = False

        if self.hparams.cfg.model.freeze_vqg:
            for param in self.vqg.parameters():
                param.requires_grad = False

    def _get_objloc_predictions(self, batch):
        pred_bbox = torch.empty((1, 6))
        pred_cls = []

        for batch_idx in range(batch["obj_conf"].shape[0]):
            scene_id = batch["scan_ids"][batch_idx]
            batch_idx_object_proposals_mask = ~batch["object_proposals_mask"][batch_idx]
            pred_bboxes = batch["pred_bboxes"]
            batch_idx_pred_bboxes = pred_bboxes[batch_idx][
                batch_idx_object_proposals_mask
            ]

            # actual prediction with highest confidence
            batch_idx_predicted_bbox_idx = torch.max(
                batch["obj_conf"][batch_idx][batch_idx_object_proposals_mask], dim=0
            ).indices
            batch_idx_pred_bbox = (
                batch_idx_pred_bboxes[batch_idx_predicted_bbox_idx].unsqueeze(0).cpu()
            )
            pred_bbox = torch.cat([pred_bbox, batch_idx_pred_bbox])

            batch_idx_pred_cls = batch["pred_cls_scores"][batch_idx][
                batch_idx_predicted_bbox_idx
            ]
            pred_cls.append(batch_idx_pred_cls)

        pred_bbox = pred_bbox[1:]

        return pred_bbox, pred_cls

    def _get_objloc_targets_and_predictions(self, batch):
        target = torch.empty((1, batch["obj_conf"].shape[1]))
        pred_bbox = torch.empty((1, 6))
        pred_cls = []

        for batch_idx in range(batch["obj_conf"].shape[0]):
            scene_id = batch["scan_ids"][batch_idx]
            batch_idx_object_proposals_mask = ~batch["object_proposals_mask"][batch_idx]
            pred_bboxes = batch["pred_bboxes"]
            batch_idx_pred_bboxes = pred_bboxes[batch_idx][
                batch_idx_object_proposals_mask
            ]

            gt_bbox = batch["gt_bbox"][batch_idx]
            gt_bbox_repeated = gt_bbox.unsqueeze(0).repeat(
                batch_idx_pred_bboxes.shape[0], 1
            )

            ious = get_ious(
                gt_bbox_repeated.cpu().numpy(), batch_idx_pred_bboxes.cpu().numpy()
            )

            # hide ious of bboxes that are background objects
            # batch_idx_cls_scores_bg = (
            #     np.array(batch["pred_iou_scores"][batch_idx]) == 18
            # ).tolist()
            # ious[batch_idx_cls_scores_bg] = 0.0

            # we take the predicted bounding box that has the highest iou to the ground truth as the bounding
            # box we need our model to give the highest confidence to
            max_idx = np.argmax(ious)

            samples = batch["obj_conf"].shape[1]
            batch_idx_target = torch.zeros(samples)
            batch_idx_target[max_idx] = 1
            batch_idx_target = batch_idx_target.unsqueeze(0).long()
            target = torch.cat([target, batch_idx_target])

            # actual prediction with highest confidence
            # print(batch["obj_conf"][batch_idx][batch_idx_object_proposals_mask])
            # print(torch.max(batch["obj_conf"][batch_idx][batch_idx_object_proposals_mask], dim=0))
            batch_idx_predicted_bbox_idx = torch.max(
                batch["obj_conf"][batch_idx][batch_idx_object_proposals_mask], dim=0
            ).indices
            batch_idx_pred_bbox = (
                batch_idx_pred_bboxes[batch_idx_predicted_bbox_idx].unsqueeze(0).cpu()
            )
            pred_bbox = torch.cat([pred_bbox, batch_idx_pred_bbox])

            batch_idx_pred_cls = batch["pred_cls_scores"][batch_idx][
                batch_idx_predicted_bbox_idx
            ]
            pred_cls.append(batch_idx_pred_cls)

        target = target[1:].long()
        pred_bbox = pred_bbox[1:]

        return target, pred_bbox, pred_cls

    def _batch_object_proposals(self, batch):
        device = batch["vert_batch_ids"].device
        softgroup_output = batch["softgroup_output"]

        object_proposals = batch["object_proposals"]
        batch_size = torch.max(batch["vert_batch_ids"]) + 1

        # get batch idx for every object proposal
        object_proposals_batch_idx = batch["vert_batch_ids"][
            softgroup_output["proposals_idx"][
                (softgroup_output["proposals_offset"][:-1]).long()
            ][:, 1].long()
        ]

        # get maximum number of object proposals in a batch
        max_object_proposals = torch.max(torch.bincount(object_proposals_batch_idx))

        # create final result with zeros
        op = torch.zeros(
            (batch_size, max_object_proposals, object_proposals.shape[1]),
            device=device,
        )

        # sort the object proposals batch idxs
        for batch_idx in range(batch_size):
            scene_id = batch["scan_ids"][batch_idx]
            batch_idx_indices = (object_proposals_batch_idx == batch_idx).nonzero(
                as_tuple=True
            )[0]
            batch_idx_object_proposals = object_proposals[batch_idx_indices.long()]
            op[batch_idx, : batch_idx_indices.shape[0]] = batch_idx_object_proposals

        return op

    def _get_predicted_bboxes(self, batch):
        device = batch["vert_batch_ids"].device
        batch_size = torch.max(batch["vert_batch_ids"]) + 1
        softgroup_output = batch["softgroup_output"]

        proposals_idx = softgroup_output["proposals_idx"]
        locs = batch["locs"]
        instances = torch.unique(proposals_idx[:, 0])
        pred_bboxes = []
        pred_bboxes_avgs = []
        for id in instances:
            proposal_idx = proposals_idx[proposals_idx[:, 0] == id]
            proposal_locs = locs[proposal_idx[:, 1].long()]
            proposal_min = proposal_locs.min(0).values
            proposal_avg = proposal_locs.mean(0)
            proposal_max = proposal_locs.max(0).values
            bbox = torch.cat([proposal_min, proposal_max])
            pred_bboxes_avgs.append(proposal_avg)
            pred_bboxes.append(bbox)
        pred_bboxes = torch.stack(pred_bboxes)
        pred_bboxes_avgs = torch.stack(pred_bboxes_avgs)

        object_proposals_batch_idx = batch["vert_batch_ids"][
            softgroup_output["proposals_idx"][
                (softgroup_output["proposals_offset"][:-1]).long()
            ][:, 1].long()
        ]

        # batch means of predicted bboxes
        max_object_proposals = torch.max(torch.bincount(object_proposals_batch_idx))
        batched_pred_bboxes_avgs = torch.zeros(
            (batch_size, max_object_proposals, pred_bboxes_avgs.shape[1]), device=device
        )
        batched_pred_bboxes = torch.zeros(
            (batch_size, max_object_proposals, pred_bboxes.shape[1]), device=device
        )
        cls_scores = []
        iou_scores = []
        for batch_idx in range(batch_size):
            scene_id = batch["scan_ids"][batch_idx]
            batch_idx_indices = (
                (object_proposals_batch_idx == batch_idx)
                .nonzero(as_tuple=True)[0]
                .long()
            )
            # predicted bboxes
            batch_idx_pred_bboxes = pred_bboxes[batch_idx_indices]
            batched_pred_bboxes[
                batch_idx, : batch_idx_indices.shape[0]
            ] = batch_idx_pred_bboxes
            # predicted bboxes classes
            batch_idx_cls_scores = (
                batch["softgroup_output"]["cls_scores"][batch_idx_indices]
                .max(dim=1)
                .indices.tolist()
            )
            batch_idx_iou_scores = (
                batch["softgroup_output"]["iou_scores"][batch_idx_indices]
                .max(dim=1)
                .indices.tolist()
            )
            cls_scores.append(batch_idx_cls_scores)
            iou_scores.append(batch_idx_iou_scores)
            # predicted bboxes center points for positional encoding
            batch_idx_pred_bboxes_avgs = pred_bboxes_avgs[batch_idx_indices]
            batched_pred_bboxes_avgs[
                batch_idx, : batch_idx_indices.shape[0]
            ] = batch_idx_pred_bboxes_avgs

        return batched_pred_bboxes, batched_pred_bboxes_avgs, cls_scores, iou_scores
