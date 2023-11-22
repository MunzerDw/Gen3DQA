import json
import os
import random

import h5py
import nltk
import numpy as np
import torch
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

from util.pc import crop
from util.transform import elastic, flip, jitter, rotz

nltk.download("punkt")

OVERFIT_NUMBER_OF_SAMPLES = 12


class ScannetQADataset(Dataset):
    def __init__(
        self, cfg, split, vocabularies, test_w_obj=False, shuffle=True, overfit=False
    ):
        self.cfg = cfg
        self.split = split
        self.shuffle = shuffle
        self.overfit = overfit
        # SCANQA
        self.test_w_obj = test_w_obj
        self.scanqa_dataset_root_path = cfg.data.scanqa.dataset_path
        self.scanqa_data_map = {
            "train": cfg.data.scanqa.splits.train_list,
            "val": cfg.data.scanqa.splits.val_list,
            "test": cfg.data.scanqa.splits.test_w_obj_list
            if test_w_obj
            else cfg.data.scanqa.splits.test_wo_obj_list,
        }
        self.tokenizer = Tokenizer(English().vocab)
        self.answer_vocabulary = vocabularies["answer_vocabulary"]
        self.answer_embeddings = vocabularies["answer_embeddings"]
        self.question_vocabulary = vocabularies["question_vocabulary"]
        self.question_embeddings = vocabularies["question_embeddings"]
        # SCANNET
        self.data_is_precomputed = False
        self.scannet_dataset_root_path = cfg.data.scannet.dataset_path
        self.file_suffix = cfg.data.scannet.file_suffix
        self.full_scale = cfg.data.scannet.full_scale
        self.scale = cfg.data.scannet.scale
        self.max_num_point = cfg.data.scannet.max_num_point
        self.scannet_data_map = {
            "train": cfg.data.scannet.metadata.train_list,
            "val": cfg.data.scannet.metadata.val_list,
            "test": cfg.data.scannet.metadata.test_list,
        }
        if cfg.model.softgroup.model.use_multiview:
            self.multiview_hdf5_file = h5py.File(
                self.cfg.data.scannet.metadata.multiview_file, "r", libver="latest"
            )
        self._load_from_disk()

    def _load_from_disk(self):
        # SCANQA
        scanqa_split = self.split
        if scanqa_split == "test":
            scanqa_split = "test_w_obj" if self.test_w_obj else "test_wo_obj"
        scanqa = json.load(
            open(
                os.path.join(
                    self.scanqa_dataset_root_path,
                    "ScanQA_v1.0_" + scanqa_split + ".json",
                )
            )
        )
        if scanqa_split == "train":
            self.scanqa = self._populate_scanqa_train_set(scanqa)
        else:
            self.scanqa = scanqa

        # for precomputing with low system memory

        # scene_ids = list(set([obj["scene_id"] for obj in self.scanqa]))
        # self.scene_ids = scene_ids
        # def chunkify(lst, n):
        #     return [lst[i::n] for i in range(n)]
        # scene_ids.sort()
        # scene_ids = chunkify(scene_ids, 4)[3]
        # print(scene_ids)

        if self.overfit:
            is_train = scanqa_split == "train"
            self.scanqa = random.sample(
                self.scanqa,
                OVERFIT_NUMBER_OF_SAMPLES if is_train else OVERFIT_NUMBER_OF_SAMPLES,
            )

        if self.shuffle:
            random.shuffle(self.scanqa)

        scene_ids = list(set([obj["scene_id"] for obj in self.scanqa]))
        self.scene_ids = scene_ids

        # tokenize question-answer pairs and get embeddings
        for qa in tqdm(self.scanqa, desc=f"Loading ScanQA {self.split} data from disk"):
            # Question
            question = qa["question"]
            tokenized_question = (
                [self.question_vocabulary["special_tokens"]["start"]]
                + self._tokenize(question)
                + [self.question_vocabulary["special_tokens"]["end"]]
            )
            tokenized_question_indicies = [
                self.question_vocabulary["token2idx"][token]
                if token in self.question_vocabulary["token2idx"].keys()
                else self.question_vocabulary["token2idx"][
                    self.question_vocabulary["special_tokens"]["unk"]
                ]
                for token in tokenized_question
            ]
            qa["tokenized_question"] = tokenized_question
            qa["question_indicies"] = tokenized_question_indicies

            # get embeddings (GLoVe)
            qa["question_embeddings"] = self.question_embeddings[
                tokenized_question_indicies
            ]
            if self.split != "test":
                # Answer
                answers = qa["answers"]
                # tokenize and get token indicies
                tokenized_answers = [
                    [self.answer_vocabulary["special_tokens"]["start"]]
                    + self._tokenize(answer)
                    + [self.answer_vocabulary["special_tokens"]["end"]]
                    for answer in answers
                ]
                tokenized_answers_indicies = [
                    [
                        self.answer_vocabulary["token2idx"][token]
                        if token in self.answer_vocabulary["token2idx"].keys()
                        else self.answer_vocabulary["token2idx"][
                            self.answer_vocabulary["special_tokens"]["unk"]
                        ]
                        for token in answer
                    ]
                    for answer in tokenized_answers
                ]
                # use only first answer as ground truth
                tokenized_answer_indicies = tokenized_answers_indicies[0]
                qa["tokenized_answers"] = tokenized_answers
                qa["answer_indicies"] = tokenized_answer_indicies

                # get embeddings (GLoVe)
                qa["answer_embeddings"] = self.answer_embeddings[
                    tokenized_answer_indicies
                ]

        # SCANNET
        precomputed_softgroup_data_path = os.path.join(
            self.cfg.data.scannet.precompute_output_path,
            f"{self.split}_scene_info.json",
        )
        if os.path.exists(precomputed_softgroup_data_path):
            # precomputed data
            self.precomputed_scenes_data = json.load(
                open(precomputed_softgroup_data_path)
            )
            self.data_is_precomputed = True
        # actual scenes
        self.scenes = {}
        for scene_name in tqdm(
            scene_ids, desc=f"Loading ScanNet {self.split} data from disk"
        ):
            # use val scenes for test_w_obj
            real_split = self.split
            if real_split == "test" and self.cfg.data.scanqa.test_w_obj:
                real_split = "val"

            scene_path = os.path.join(
                self.scannet_dataset_root_path,
                real_split,
                scene_name + self.file_suffix,
            )
            scene = torch.load(scene_path)
            scene["xyz"] -= scene["xyz"].mean(axis=0)
            scene["rgb"] = scene["rgb"].astype(np.float32) / 127.5 - 1
            self.scenes[scene_name] = scene

    def __len__(self):
        if self.cfg.model.precompute_softgroup_data:
            return len(self.scene_ids)
        return len(self.scanqa)

    def __getitem__(self, idx):
        if self.cfg.model.precompute_softgroup_data:
            return self._get_scene_item(idx)

        # SCANQA
        qa = self.scanqa[idx]
        data = qa.copy()

        # augment question
        if self.split == "train" and not self.cfg.model.freeze_vqa:
            # copy embeddings
            data["question_embeddings"] = data["question_embeddings"].copy()
            question_indices = data["question_indicies"][1:-1]

            # randomly decide how many words to augment (replace with <unk> embedding)
            rand_n_to_augment = random.randrange(0, 1)

            if rand_n_to_augment != 0:
                # get the random indices to augment
                rand_idxs = random.sample(range(len(question_indices)), rand_n_to_augment)

                # augment with <unk> embedding
                token2idx = self.question_vocabulary["token2idx"]
                question_embeddings = self.question_embeddings
                unk_emb = question_embeddings[token2idx["<unk>"]]
                data["question_embeddings"][rand_idxs] = unk_emb

        # SCANNET
        scene_id = qa["scene_id"]
        data["scan_id"] = scene_id
        scene = self.scenes[scene_id]

        if self.data_is_precomputed:
            precomputed_scene_data = self.precomputed_scenes_data[scene_id]
            data["pred_bboxes"] = precomputed_scene_data["pred_bboxes"]
            data["pred_bboxes_avgs"] = precomputed_scene_data["pred_bboxes_avgs"]
            data["pred_cls_scores"] = precomputed_scene_data["pred_cls_scores"]
            data["pred_iou_scores"] = precomputed_scene_data["pred_iou_scores"]
            data["object_proposals"] = precomputed_scene_data["object_proposals"]

        points = scene["xyz"]  # (N, 3)
        colors = scene["rgb"]  # (N, 3)
        normals = scene["normal"]
        if self.cfg.model.softgroup.model.use_multiview:
            multiviews = self.multiview_hdf5_file[scene_id]
        instance_ids = scene["instance_ids"]
        sem_labels = scene["sem_labels"]

        # augment
        if self.split == "train" and not self.cfg.model.freeze_softgroup:
            aug_matrix = self._get_augmentation_matrix()
            points = np.matmul(points, aug_matrix)
            normals = np.matmul(normals, np.transpose(np.linalg.inv(aug_matrix)))
            if self.cfg.data.scannet.augmentation.jitter_rgb:
                # jitter rgb
                colors += np.random.randn(3) * 0.1

        # scale
        scaled_points = points * self.scale

        # elastic
        if (
            self.split == "train"
            and self.cfg.data.scannet.augmentation.elastic
            and not self.cfg.model.freeze_softgroup
        ):
            scaled_points = elastic(
                scaled_points, 6 * self.scale // 50, 40 * self.scale / 50
            )
            scaled_points = elastic(
                scaled_points, 20 * self.scale // 50, 160 * self.scale / 50
            )

        # offset
        scaled_points -= scaled_points.min(axis=0)

        # crop
        if self.split == "train" and not self.cfg.model.freeze_softgroup:
            # HACK, in case there are few points left
            max_tries = 10
            valid_idxs_count = 0
            valid_idxs = np.ones(shape=scaled_points.shape[0], dtype=np.bool)
            if valid_idxs.shape[0] > self.max_num_point:
                while max_tries > 0:
                    points_tmp, valid_idxs = crop(
                        scaled_points, self.max_num_point, self.full_scale[1]
                    )
                    valid_idxs_count = np.count_nonzero(valid_idxs)
                    if valid_idxs_count >= 5000:
                        scaled_points = points_tmp
                        break
                    max_tries -= 1
                if valid_idxs_count < 5000:
                    raise Exception("Over-cropped!")

            scaled_points = scaled_points[valid_idxs]
            points = points[valid_idxs]
            normals = normals[valid_idxs]
            colors = colors[valid_idxs]
            if self.cfg.model.softgroup.model.use_multiview:
                multiviews = np.asarray(multiviews)[valid_idxs]
            sem_labels = sem_labels[valid_idxs]
            instance_ids = self._get_cropped_inst_ids(instance_ids, valid_idxs)

        (
            num_instance,
            instance_info,
            instance_num_point,
            instance_semantic_cls,
        ) = self._get_inst_info(points, instance_ids, sem_labels)

        feats = np.zeros(shape=(len(scaled_points), 0), dtype=np.float32)
        if self.cfg.model.softgroup.model.use_color:
            feats = np.concatenate((feats, colors), axis=1)
        if self.cfg.model.softgroup.model.use_normal:
            feats = np.concatenate((feats, normals), axis=1)
        if self.cfg.model.softgroup.model.use_multiview:
            feats = np.concatenate((feats, multiviews), axis=1)

        data["locs"] = points  # (N, 3)
        data["locs_scaled"] = scaled_points  # (N, 3)
        data["feats"] = feats  # (N, 3)
        data["sem_labels"] = sem_labels  # (N,)
        data["instance_ids"] = instance_ids  # (N,) 0~total_nInst, -1
        data["num_instance"] = np.array(num_instance, dtype=np.int32)  # int
        data["instance_info"] = instance_info  # (N, 12)
        data["instance_num_point"] = np.array(
            instance_num_point, dtype=np.int32
        )  # (num_instance,)
        data["instance_semantic_cls"] = instance_semantic_cls
        if self.split != "test":
            # gt bbox
            data["realobjids"] = scene["realobjids"]
            inst_id = scene["realobjids"]["objids2indids"][qa["object_ids"][0]]
            data["aligned_gt_bbox"] = scene["aligned_instance_bboxes"][inst_id]
            gt_bbox_label, gt_bbox_instance_id, gt_bbox = self._get_gt_bbox(
                points,
                instance_ids,
                sem_labels,
                self.cfg.data.scannet.ignore_classes,
                inst_id,
            )
            data["gt_bbox"] = gt_bbox
            data["gt_bbox_label"] = gt_bbox_label
            data["gt_bbox_instance_id"] = gt_bbox_instance_id
            
            data["gt_bboxes"] = []
            for object_id in qa["object_ids"]:
                inst_id = scene["realobjids"]["objids2indids"][object_id]
                gt_bbox_label, gt_bbox_instance_id, gt_bbox = self._get_gt_bbox(
                    points,
                    instance_ids,
                    sem_labels,
                    self.cfg.data.scannet.ignore_classes,
                    inst_id,
                )
                data["gt_bboxes"].append(gt_bbox)
            
        return data

    # SCANNET

    def _get_scene_item(self, idx):
        # SCANNET
        scene_id = self.scene_ids[idx]
        scene = self.scenes[scene_id]

        points = scene["xyz"]  # (N, 3)
        colors = scene["rgb"]  # (N, 3)
        normals = scene["normal"]
        instance_ids = scene["instance_ids"]
        sem_labels = scene["sem_labels"]
        data = {"scan_id": scene_id}

        # scale
        scaled_points = points * self.scale

        # offset
        scaled_points -= scaled_points.min(axis=0)

        (
            num_instance,
            instance_info,
            instance_num_point,
            instance_semantic_cls,
        ) = self._get_inst_info(points, instance_ids, sem_labels)

        feats = np.zeros(shape=(len(scaled_points), 0), dtype=np.float32)
        if self.cfg.model.softgroup.model.use_color:
            feats = np.concatenate((feats, colors), axis=1)
        if self.cfg.model.softgroup.model.use_normal:
            feats = np.concatenate((feats, normals), axis=1)

        data["locs"] = points  # (N, 3)
        data["locs_scaled"] = scaled_points  # (N, 3)
        data["feats"] = feats  # (N, 3)
        data["sem_labels"] = sem_labels  # (N,)
        data["instance_ids"] = instance_ids  # (N,) 0~total_nInst, -1
        data["num_instance"] = np.array(num_instance, dtype=np.int32)  # int
        data["instance_info"] = instance_info  # (N, 12)
        data["instance_num_point"] = np.array(
            instance_num_point, dtype=np.int32
        )  # (num_instance,)
        data["instance_semantic_cls"] = instance_semantic_cls
        data["realobjids"] = scene["realobjids"]

        return data

    def _get_gt_bbox(self, xyz, instance_ids, sem_labels, ignore_classes, instance_id):
        idx = instance_ids == instance_id
        sem_label = sem_labels[idx][0]
        sem_label = sem_label - len(ignore_classes)

        xyz_i = xyz[idx]
        min_xyz = xyz_i.min(0)
        max_xyz = xyz_i.max(0)

        return sem_label, instance_id, np.concatenate((min_xyz, max_xyz))

    def _get_augmentation_matrix(self):
        m = np.eye(3)
        if self.cfg.data.scannet.augmentation.jitter_xyz:
            m = np.matmul(m, jitter())
        if self.cfg.data.scannet.augmentation.flip:
            flip_m = flip(0, random=True)
            m *= flip_m
        if self.cfg.data.scannet.augmentation.rotation:
            t = np.random.rand() * 2 * np.pi
            rot_m = rotz(t)
            m = np.matmul(m, rot_m)  # rotation around z
        return m.astype(np.float32)

    def _get_cropped_inst_ids(self, instance_ids, valid_idxs):
        """
        Postprocess instance_ids after cropping
        """
        instance_ids = instance_ids[valid_idxs]
        j = 0
        while j < instance_ids.max():
            if np.count_nonzero(instance_ids == j) == 0:
                instance_ids[instance_ids == instance_ids.max()] = j
            j += 1
        return instance_ids

    def _get_inst_info(self, xyz, instance_ids, sem_labels):
        """
        :param xyz: (n, 3)
        :param instance_ids: (n), int, (0~nInst-1, -1)
        :return: num_instance, dict
        """
        instance_num_point = []  # (nInst), int
        unique_instance_ids = np.unique(instance_ids)
        unique_instance_ids = unique_instance_ids[
            unique_instance_ids != self.cfg.data.scannet.ignore_label
        ]
        num_instance = unique_instance_ids.shape[0]
        # (n, 3), float, (meanx, meany, meanz)
        instance_info = np.empty(shape=(xyz.shape[0], 3), dtype=np.float32)
        instance_cls = np.full(
            shape=unique_instance_ids.shape[0],
            fill_value=self.cfg.data.scannet.ignore_label,
            dtype=np.int8,
        )
        for index, i in enumerate(unique_instance_ids):
            inst_i_idx = np.where(instance_ids == i)[0]

            # instance_info
            xyz_i = xyz[inst_i_idx]

            mean_xyz_i = xyz_i.mean(0)

            # offset
            instance_info[inst_i_idx] = mean_xyz_i

            # instance_num_point
            instance_num_point.append(inst_i_idx.size)

            # semantic label
            cls_idx = inst_i_idx[0]
            instance_cls[index] = (
                sem_labels[cls_idx] - len(self.cfg.data.scannet.ignore_classes)
                if sem_labels[cls_idx] != self.cfg.data.scannet.ignore_label
                else sem_labels[cls_idx]
            )
            # bounding boxes

        return num_instance, instance_info, instance_num_point, instance_cls

    # SCANQA

    def _populate_scanqa_train_set(self, scanqa):
        for sample in scanqa:
            if len(sample["answers"]) > 1:
                for n, answer in enumerate(sample["answers"][1:]):
                    copy = sample.copy()
                    copy["answers"] = [answer]
                    copy["question_id"] = copy["question_id"] + "-" + str(n)
                    scanqa.append(copy)

        return scanqa

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
