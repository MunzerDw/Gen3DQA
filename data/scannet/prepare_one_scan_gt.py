'''
Generate instance groundtruth .txt files (for evaluation)
'''

import torch
import os
import argparse
import numpy as np
from omegaconf import OmegaConf

semantic_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
semantic_label_names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id', help='scan id', required=True)
    parser.add_argument('-s', '--split', help='data split (train / val / test)', default='val')
    parser.add_argument('-c', '--cfg', help='scannet configuration YAML file', default='../../conf/path.yaml')
    opt = parser.parse_args()

    cfg = OmegaConf.load(opt.cfg)
    cfg.split = opt.split

    root = os.path.join(cfg.SCANNETV2_PATH.splited_data, cfg.split)

    scene_name = opt.id
    file_path = os.path.join(root, f"{scene_name}.pth")
    room = torch.load(file_path)
    os.makedirs(os.path.join(cfg.SCANNETV2_PATH.splited_gt, cfg.split), exist_ok=True)


    instance_ids = room["instance_ids"]
    label = room["sem_labels"]  # {-1,0,1,...,19}

    # xyz, rgb, label, instance_ids = rooms[i]   # label 0~19 -1;  instance_ids 0~instance_num-1 -1

    # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)
    # instance_ids_new = np.zeros(instance_ids.shape, dtype=np.int32)
    instance_ids_new = (label.copy().astype(np.int32) + 1) * 1000

    # instance_num = int(instance_ids.max()) + 1
    # for inst_id in range(instance_num):
    unique_instance_ids = np.unique(instance_ids)
    for inst_id in unique_instance_ids:
        if inst_id < 0: continue

        instance_mask = np.where(instance_ids == inst_id)[0]
        sem_id = int(label[instance_mask[0]])
        if(sem_id == -1):
            semantic_label = 0
        else:
            semantic_label = semantic_label_idxs[sem_id]
        instance_ids_new[instance_mask] = semantic_label * 1000 + inst_id + 1
        # instance_ids_new[instance_mask] += inst_id + 1

    np.savetxt(os.path.join(cfg.SCANNETV2_PATH.splited_gt, cfg.split, "{}.txt".format(scene_name)), instance_ids_new, fmt='%d')