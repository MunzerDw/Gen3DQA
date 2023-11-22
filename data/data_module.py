import collections
import json
import os
import pickle
from importlib import import_module

import numpy as np
import pytorch_lightning as pl
import torch
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from torch.utils.data import DataLoader

from common_ops.functions import common_ops


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset = getattr(
            import_module("data.dataset.scannetqa_dataset"), "ScannetQADataset"
        )
        self.tokenizer = Tokenizer(English().vocab)
        self.vocabularies = self._get_vocabulary()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = self.dataset(
                self.cfg,
                "train",
                self.vocabularies,
                shuffle=self.cfg.data.shuffle,
                overfit=self.cfg.data.overfit,
            )
            self.val_set = self.dataset(
                self.cfg,
                "val",
                self.vocabularies,
                shuffle=self.cfg.data.shuffle,
                overfit=self.cfg.data.overfit,
            )
        if stage == "test" or stage is None:
            self.val_set = self.dataset(
                self.cfg,
                self.cfg.model.inference.split,
                self.vocabularies,
                shuffle=self.cfg.data.shuffle,
                overfit=self.cfg.data.overfit,
            )
        if stage == "predict" or stage is None:
            self.test_set = self.dataset(
                self.cfg,
                "test",
                self.vocabularies,
                test_w_obj=self.cfg.data.scanqa.test_w_obj,
                shuffle=self.cfg.data.shuffle,
                overfit=self.cfg.data.overfit,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=sparse_collate_fn if self.cfg.model.precompute_softgroup_data == False else sparse_collate_fn_softgroup,
            num_workers=self.cfg.data.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.cfg.data.batch_size,
            pin_memory=True,
            collate_fn=sparse_collate_fn if self.cfg.model.precompute_softgroup_data == False else sparse_collate_fn_softgroup,
            num_workers=self.cfg.data.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            pin_memory=True,
            collate_fn=sparse_collate_fn,
            num_workers=self.cfg.data.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=1,
            pin_memory=True,
            collate_fn=sparse_collate_fn_predicting,
            num_workers=self.cfg.data.num_workers,
        )

    def _get_vocabulary(self):
        """
        :return: dict - vocabularies and embeddings
        """
        # build 2 separate vocabularies for answer and question
        print(f"Building ScanQA vocabulary...")
        scanqa_train = json.load(
            open(
                os.path.join(
                    self.cfg.data.scanqa.dataset_path, "ScanQA_v1.0_train.json"
                )
            )
        )

        # answer vocab
        answers = sum([data["answers"] for data in scanqa_train], [])
        answer_vocabulary, answer_embeddings = self._build_vocabulary(
            self.cfg.data.dataset_root_path, answers, vocab_name="answer"
        )

        # question vocab
        questions = [data["question"] for data in scanqa_train]
        question_vocabulary, question_embeddings = self._build_vocabulary(
            self.cfg.data.dataset_root_path, questions, vocab_name="question"
        )

        return {
            "answer_vocabulary": answer_vocabulary,
            "answer_embeddings": answer_embeddings,
            "question_vocabulary": question_vocabulary,
            "question_embeddings": question_embeddings,
        }

    def _tokenize(self, sentence):
        """
        :param sentences: (1), string
        :return: (n), list of tokens (strings)
        """
        sentence = sentence.replace("?", " ?")
        sentence = sentence.replace(",", " ,")
        sentence = sentence.replace("(", "( ")
        sentence = sentence.replace(")", " )")
        return [token.text.lower() for token in self.tokenizer(sentence)]

    def _build_vocabulary(self, dataset_root_path, sentences, vocab_name):
        """
        :param sentences: (n), strings
        :return: dict, npy - vocabulary with token2idx, idx2token, and special_tokens
        """
        # load vocabulary if exists
        vocab_path = os.path.join(
            dataset_root_path, "vocabularies", vocab_name + "_vocabulary.json"
        )
        if os.path.exists(vocab_path):
            vocabulary = json.load(open(vocab_path))
        else:
            token2idx, idx2token = {}, {}
            special_tokens = ["<pad>", "<unk>", "<start>", "<end>", "<objloc>"]

            # tokenize + count all tokens
            tokens_counter = collections.Counter(
                sum([self._tokenize(sentence) for sentence in sentences], [])
            )

            # get sortedtokens list
            tokens = sorted(tokens_counter.keys())

            # store tokens in dictionaries
            for i, token in enumerate(tokens):
                index = i + len(special_tokens)
                token2idx[token] = index
                idx2token[index] = token

            # store special tokens in dictionaries
            for i, special_token in enumerate(special_tokens):
                token2idx[special_token] = i
                idx2token[i] = special_token

            vocabulary = {
                "token2idx": token2idx,
                "idx2token": idx2token,
                "special_tokens": {
                    "pad": "<pad>",
                    "unk": "<unk>",
                    "start": "<start>",
                    "end": "<end>",
                    "objloc": "<objloc>",
                },
            }

            # write vocabulary to load faster in the future
            json.dump(vocabulary, open(vocab_path, "w"), indent=4)

        # glove
        glove_path = os.path.join(dataset_root_path, "scanqa", "glove.p")
        with open(glove_path, "rb") as f:
            glove = pickle.load(f)

        # special tokens embeddings
        st_emb_path = os.path.join(
            dataset_root_path, "vocabularies", "special_tokens_embeddings.npy"
        )
        if os.path.exists(st_emb_path):
            st_embeddings = np.load(st_emb_path)
        else:
            st_embeddings = np.zeros((len(vocabulary["special_tokens"]), 300))

            for idx, token in enumerate(special_tokens):
                if token == vocabulary["special_tokens"]["pad"]:
                    emb = np.zeros(300)
                elif token == vocabulary["special_tokens"]["unk"]:
                    emb = glove["unk"]
                else:
                    emb = np.random.rand(300)
                st_embeddings[idx] = emb

            np.save(st_emb_path, st_embeddings)

        # embeddings
        emb_path = os.path.join(
            dataset_root_path, "vocabularies", vocab_name + "_embeddings.npy"
        )
        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)
        else:
            embeddings = np.zeros((len(vocabulary["token2idx"]), 300))
            for token, idx in vocabulary["token2idx"].items():
                if token in special_tokens:
                    emb = st_embeddings[special_tokens.index(token)]
                elif token in glove:
                    emb = glove[token]
                else:
                    emb = glove["unk"]
                embeddings[idx] = emb

            np.save(emb_path, embeddings)

        # casting to ints
        vocabulary["idx2token"] = {
            int(k): v for k, v in vocabulary["idx2token"].items()
        }
        vocabulary["token2idx"] = {
            k: int(v) for k, v in vocabulary["token2idx"].items()
        }

        return vocabulary, embeddings


def sparse_collate_fn(batch):
    data = {}

    # SCANNET

    locs = []
    locs_scaled = []
    vert_batch_ids = []
    feats = []
    sem_labels = []
    instance_ids = []
    instance_info = []  # (N, 3)
    instance_num_point = []  # (total_nInst), int
    instance_offsets = [0]
    total_num_inst = 0
    instance_cls = []  # (total_nInst), long
    scan_ids = []
    num_instance = []
    realobjids = []
    gt_bboxes = []
    gt_bbox = []
    gt_bbox_label = []
    gt_bbox_instance_id = []
    # precomputed data
    pred_bboxes = []
    pred_bboxes_avgs = []
    pred_cls_scores = []
    pred_iou_scores = []
    object_proposals = []
    if "object_proposals" in batch[0]:
        max_object_proposals = np.max([len(b["object_proposals"]) for b in batch])

    # SCANQA

    questions = []
    answers = []
    object_ids = []
    object_names = []
    question_id = []
    tokenized_answers = []
    tokenized_question = []
    answer_indicies = []
    question_indicies = []
    answer_embeddings = []
    question_embeddings = []
    max_question_length = np.max([len(b["question_indicies"]) for b in batch])
    max_answer_length = np.max([len(b["answer_indicies"]) for b in batch])

    for i, b in enumerate(batch):

        # SCANNET

        scan_ids.append(b["scan_id"])
        realobjids.append(b["realobjids"])
        gt_bboxes.append(b["gt_bboxes"])
        gt_bbox.append(torch.from_numpy(b["gt_bbox"]).unsqueeze(0))
        gt_bbox_label.append(b["gt_bbox_label"])
        gt_bbox_instance_id.append(b["gt_bbox_instance_id"])
        locs.append(torch.from_numpy(b["locs"]))

        locs_scaled.append(torch.from_numpy(b["locs_scaled"]).int())
        vert_batch_ids.append(
            torch.full((b["locs_scaled"].shape[0],), fill_value=i, dtype=torch.int16)
        )
        feats.append(torch.from_numpy(b["feats"]))

        instance_ids_i = b["instance_ids"].copy()
        instance_ids_i[instance_ids_i != -1] += total_num_inst
        total_num_inst += b["num_instance"].item()
        instance_ids.append(torch.from_numpy(instance_ids_i))

        sem_labels.append(torch.from_numpy(b["sem_labels"]))

        instance_info.append(torch.from_numpy(b["instance_info"]))
        instance_num_point.append(torch.from_numpy(b["instance_num_point"]))
        instance_offsets.append(instance_offsets[-1] + b["num_instance"].item())

        instance_cls.extend(b["instance_semantic_cls"])

        num_instance.append(b["num_instance"])

        # precomputed data
        if "object_proposals" in b:
            pred_cls_scores.append(b["pred_cls_scores"])
            pred_iou_scores.append(b["pred_iou_scores"])
            # pad pred bboxes
            pred_bboxes_b = torch.cat(
                [
                    torch.tensor(b["pred_bboxes"]),
                    torch.zeros(
                        (
                            max_object_proposals - len(b["pred_bboxes"]),
                            len(b["pred_bboxes"][0]),
                        )
                    ),
                ]
            )
            pred_bboxes.append(pred_bboxes_b)
            # pad averages of pred bboxes
            pred_bboxes_avgs_b = torch.cat(
                [
                    torch.tensor(b["pred_bboxes_avgs"]),
                    torch.zeros(
                        (
                            max_object_proposals - len(b["pred_bboxes_avgs"]),
                            len(b["pred_bboxes_avgs"][0]),
                        )
                    ),
                ]
            )
            pred_bboxes_avgs.append(pred_bboxes_avgs_b)
            # pad object proposals
            object_proposals_b = torch.cat(
                [
                    torch.tensor(b["object_proposals"]),
                    torch.zeros(
                        (
                            max_object_proposals - len(b["object_proposals"]),
                            len(b["object_proposals"][0]),
                        )
                    ),
                ]
            )
            object_proposals.append(object_proposals_b)

        # SCANQA

        questions.append(b["question"])
        answers.append(b["answers"])
        object_ids.append(b["object_ids"])
        object_names.append(b["object_names"])
        question_id.append(b["question_id"])
        tokenized_answers.append(b["tokenized_answers"])
        tokenized_question.append(b["tokenized_question"])
        # pad answer indicies for XE loss later
        a = (
            b["answer_indicies"] + (max_answer_length - len(b["answer_indicies"])) * [0]
        )[:max_answer_length]
        answer_indicies.append(a)
        # pad question indicies for XE loss later
        a = (
            b["question_indicies"] + (max_question_length - len(b["question_indicies"])) * [0]
        )[:max_question_length]
        question_indicies.append(a)
        # pad answer embeddings
        answer_embeddings_b = torch.cat(
            [
                torch.from_numpy(b["answer_embeddings"]),
                torch.zeros(
                    (
                        max_answer_length - len(b["answer_embeddings"]),
                        b["answer_embeddings"].shape[1],
                    )
                ),
            ]
        )
        answer_embeddings.append(answer_embeddings_b)
        # pad question embeddings
        # remove <start> and <end> tokens
        question_embeddings_b = b["question_embeddings"]
        question_embeddings_b = torch.cat(
            [
                torch.from_numpy(question_embeddings_b),
                torch.zeros(
                    (
                        max_question_length - len(question_embeddings_b),
                        b["question_embeddings"].shape[1],
                    )
                ),
            ]
        )
        question_embeddings.append(question_embeddings_b)

    # SCANNET

    data["scan_ids"] = scan_ids
    data["realobjids"] = realobjids
    data["gt_bboxes"] = gt_bboxes
    data["gt_bbox"] = torch.cat(gt_bbox, dim=0)  # float (B, 6)
    data["gt_bbox_label"] = torch.tensor(gt_bbox_label)  # (B, 1)
    data["gt_bbox_instance_id"] = torch.tensor(gt_bbox_instance_id)  # (B, 1)
    data["locs"] = torch.cat(locs, dim=0)  # float (N, 3)
    data["vert_batch_ids"] = torch.cat(vert_batch_ids, dim=0)
    data["feats"] = torch.cat(feats, dim=0)

    data["sem_labels"] = torch.cat(sem_labels, dim=0)  # int (N,)
    data["instance_ids"] = torch.cat(instance_ids, dim=0)  # int, (N,)
    data["instance_info"] = torch.cat(instance_info, dim=0)  # float (total_nInst, 3)
    data["instance_num_point"] = torch.cat(instance_num_point, dim=0)  # (total_nInst)
    data["instance_offsets"] = torch.tensor(
        instance_offsets, dtype=torch.int32
    )  # int (B+1)
    data["instance_semantic_cls"] = torch.tensor(
        instance_cls, dtype=torch.int32
    )  # long (total_nInst)
    data["num_instance"] = num_instance

    # voxelize
    # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    tmp_locs_scaled = torch.cat(locs_scaled, dim=0)
    data["voxel_locs"], data["v2p_map"], data["p2v_map"] = common_ops.voxelization_idx(
        tmp_locs_scaled, data["vert_batch_ids"], len(batch), 4
    )

    # precomputed data
    if len(pred_bboxes) > 0:
        data["pred_bboxes"] = torch.stack(pred_bboxes)
        data["object_proposals"] = torch.stack(object_proposals)
        data["pred_bboxes_avgs"] = torch.stack(pred_bboxes_avgs)
        data["pred_cls_scores"] = pred_cls_scores
        data["pred_iou_scores"] = pred_iou_scores

    # SCANQA

    data["question"] = questions
    data["answers"] = answers
    data["object_ids"] = object_ids
    data["object_names"] = object_names
    data["question_id"] = question_id
    data["tokenized_answers"] = tokenized_answers
    data["tokenized_question"] = tokenized_question
    data["answer_indicies"] = torch.Tensor(answer_indicies).long()
    data["question_indicies"] = torch.Tensor(question_indicies).long()
    data["answer_embeddings"] = torch.stack(answer_embeddings)
    data["question_embeddings"] = torch.stack(question_embeddings)

    return data


def sparse_collate_fn_predicting(batch):
    data = {}

    # SCANNET

    locs = []
    locs_scaled = []
    vert_batch_ids = []
    feats = []
    sem_labels = []
    instance_ids = []
    instance_info = []  # (N, 3)
    instance_num_point = []  # (total_nInst), int
    instance_offsets = [0]
    total_num_inst = 0
    instance_cls = []  # (total_nInst), long
    scan_ids = []
    num_instance = []

    # SCANQA

    questions = []
    question_id = []
    tokenized_question = []
    question_indicies = []
    question_embeddings = []
    max_question_length = np.max([len(b["question_indicies"]) for b in batch])

    for i, b in enumerate(batch):

        # SCANNET

        scan_ids.append(b["scan_id"])
        locs.append(torch.from_numpy(b["locs"]))

        locs_scaled.append(torch.from_numpy(b["locs_scaled"]).int())
        vert_batch_ids.append(
            torch.full((b["locs_scaled"].shape[0],), fill_value=i, dtype=torch.int16)
        )
        feats.append(torch.from_numpy(b["feats"]))

        instance_ids_i = b["instance_ids"].copy()
        instance_ids_i[instance_ids_i != -1] += total_num_inst
        total_num_inst += b["num_instance"].item()
        instance_ids.append(torch.from_numpy(instance_ids_i))

        sem_labels.append(torch.from_numpy(b["sem_labels"]))

        instance_info.append(torch.from_numpy(b["instance_info"]))
        instance_num_point.append(torch.from_numpy(b["instance_num_point"]))
        instance_offsets.append(instance_offsets[-1] + b["num_instance"].item())

        instance_cls.extend(b["instance_semantic_cls"])

        num_instance.append(b["num_instance"])

        # SCANQA

        questions.append(b["question"])
        question_id.append(b["question_id"])
        tokenized_question.append(b["tokenized_question"])
        question_indicies.append(b["question_indicies"])
        # pad question embeddings
        # remove <start> and <end> tokens
        question_embeddings_b = b["question_embeddings"]
        question_embeddings_b = torch.cat(
            [
                torch.from_numpy(question_embeddings_b),
                torch.zeros(
                    (
                        max_question_length - len(question_embeddings_b),
                        b["question_embeddings"].shape[1],
                    )
                ),
            ]
        )
        question_embeddings.append(question_embeddings_b)

    # SCANNET

    data["scan_ids"] = scan_ids
    data["locs"] = torch.cat(locs, dim=0)  # float (N, 3)
    data["vert_batch_ids"] = torch.cat(vert_batch_ids, dim=0)
    data["feats"] = torch.cat(feats, dim=0)

    data["sem_labels"] = torch.cat(sem_labels, dim=0)  # int (N,)
    data["instance_ids"] = torch.cat(instance_ids, dim=0)  # int, (N,)
    data["instance_info"] = torch.cat(instance_info, dim=0)  # float (total_nInst, 3)
    data["instance_num_point"] = torch.cat(instance_num_point, dim=0)  # (total_nInst)
    data["instance_offsets"] = torch.tensor(
        instance_offsets, dtype=torch.int32
    )  # int (B+1)
    data["instance_semantic_cls"] = torch.tensor(
        instance_cls, dtype=torch.int32
    )  # long (total_nInst)
    data["num_instance"] = num_instance

    # voxelize
    # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    tmp_locs_scaled = torch.cat(locs_scaled, dim=0)
    data["voxel_locs"], data["v2p_map"], data["p2v_map"] = common_ops.voxelization_idx(
        tmp_locs_scaled, data["vert_batch_ids"], len(batch), 4
    )

    # SCANQA

    data["question"] = questions
    data["question_id"] = question_id
    data["tokenized_question"] = tokenized_question
    data["question_indicies"] = question_indicies
    data["question_embeddings"] = torch.stack(question_embeddings)

    return data


def sparse_collate_fn_softgroup(batch):
    data = {}
    locs = []
    locs_scaled = []
    vert_batch_ids = []
    feats = []
    sem_labels = []
    instance_ids = []
    instance_info = []  # (N, 3)
    instance_num_point = []  # (total_nInst), int
    instance_offsets = [0]
    total_num_inst = 0
    instance_cls = []  # (total_nInst), long

    scan_ids = []

    for i, b in enumerate(batch):
        scan_ids.append(b["scan_id"])
        locs.append(torch.from_numpy(b["locs"]))

        locs_scaled.append(torch.from_numpy(b["locs_scaled"]).int())
        vert_batch_ids.append(
            torch.full((b["locs_scaled"].shape[0],), fill_value=i, dtype=torch.int16)
        )
        feats.append(torch.from_numpy(b["feats"]))

        instance_ids_i = b["instance_ids"]
        instance_ids_i[instance_ids_i != -1] += total_num_inst
        total_num_inst += b["num_instance"].item()
        instance_ids.append(torch.from_numpy(instance_ids_i))

        sem_labels.append(torch.from_numpy(b["sem_labels"]))

        instance_info.append(torch.from_numpy(b["instance_info"]))
        instance_num_point.append(torch.from_numpy(b["instance_num_point"]))
        instance_offsets.append(instance_offsets[-1] + b["num_instance"].item())

        instance_cls.extend(b["instance_semantic_cls"])

    tmp_locs_scaled = torch.cat(
        locs_scaled, dim=0
    )  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    data["scan_ids"] = scan_ids
    data["locs"] = torch.cat(locs, dim=0)  # float (N, 3)
    data["vert_batch_ids"] = torch.cat(vert_batch_ids, dim=0)
    data["feats"] = torch.cat(feats, dim=0)

    data["sem_labels"] = torch.cat(sem_labels, dim=0)  # int (N,)
    data["instance_ids"] = torch.cat(instance_ids, dim=0)  # int, (N,)
    data["instance_info"] = torch.cat(instance_info, dim=0)  # float (total_nInst, 3)
    data["instance_num_point"] = torch.cat(instance_num_point, dim=0)  # (total_nInst)
    data["instance_offsets"] = torch.tensor(
        instance_offsets, dtype=torch.int32
    )  # int (B+1)
    data["instance_semantic_cls"] = torch.tensor(
        instance_cls, dtype=torch.int32
    )  # long (total_nInst)

    # voxelize
    data["voxel_locs"], data["v2p_map"], data["p2v_map"] = common_ops.voxelization_idx(
        tmp_locs_scaled, data["vert_batch_ids"], len(batch), 4
    )
    return data
