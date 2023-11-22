"""
REFERENCE TO https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
"""

import json
import os
from functools import partial

import hydra
import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData
from tqdm.contrib.concurrent import process_map

IGNORE_CLASS_IDS = np.array([1, 2, 22])  # exclude wall, floor and ceiling

LABEL_MAP_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "metadata/scannetv2-labels.combined.tsv",
)
G_LABEL_NAMES = [
    "unannotated",
    "wall",
    "floor",
    "chair",
    "table",
    "desk",
    "bed",
    "bookshelf",
    "sofa",
    "sink",
    "bathtub",
    "toilet",
    "curtain",
    "counter",
    "door",
    "window",
    "shower curtain",
    "refrigerator",
    "picture",
    "cabinet",
    "others",
]
type2class = {
    "wall": 0,
    "floor": 1,
    "cabinet": 2,
    "bed": 3,
    "chair": 4,
    "sofa": 5,
    "table": 6,
    "door": 7,
    "window": 8,
    "bookshelf": 9,
    "picture": 10,
    "counter": 11,
    "desk": 12,
    "curtain": 13,
    "refrigerator": 14,
    "shower curtain": 15,
    "toilet": 16,
    "sink": 17,
    "bathtub": 18,
    "others": 19,
}
nyu40ids = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
]

remapper = np.full(shape=150, fill_value=-1, dtype=np.int32)


def get_raw2scannetv2_label_map():
    lines = [line.rstrip() for line in open(LABEL_MAP_FILE)]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(G_LABEL_NAMES)
        elements = lines[i].split("\t")
        raw_name = elements[1]
        nyu40_id = int(elements[4])
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            if nyu40_id in nyu40ids:
                raw2scannet[raw_name] = "others"
                remapper[nyu40_id] = type2class["others"]
            else:
                raw2scannet[raw_name] = "unannotated"
        else:
            raw2scannet[raw_name] = nyu40_name
            remapper[nyu40_id] = type2class[nyu40_name]
    return raw2scannet


# Map relevant classes to {0,1,...,19}, and ignored classes to -1
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
OBJECT_MAPPING = get_raw2scannetv2_label_map()


def read_mesh_file(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.rint(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8),
        np.asarray(mesh.vertex_normals, dtype=np.float32),
    )


def read_axis_align_matrix(meta_file):
    axis_align_matrix = None
    with open(meta_file, "r") as f:
        for line in f:
            line_content = line.strip()
            if "axisAlignment" in line_content:
                axis_align_matrix = [
                    float(x) for x in line_content.strip("axisAlignment = ").split(" ")
                ]
                axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
                break
    return axis_align_matrix


def read_label_file(label_file):
    plydata = PlyData.read(label_file)
    sem_labels = np.array(plydata["vertex"]["label"], dtype=np.int32)  # nyu40
    return sem_labels


def read_agg_file(agg_file):
    object_id2segs = {}
    label2segs = {}
    realobjids = {}
    realobjids["indids2objids"] = {}
    realobjids["objids2indids"] = {}
    object_id = 0
    with open(agg_file, "r") as json_data:
        data = json.load(json_data)
        for group in data["segGroups"]:
            label = group["label"]
            segs = group["segments"]
            if OBJECT_MAPPING[label] not in ["wall", "floor", "ceiling", "unannotated"]:
                object_id2segs[object_id] = segs
                if label in label2segs:
                    label2segs[label].extend(segs)
                else:
                    label2segs[label] = segs.copy()
                realobjids["indids2objids"][object_id] = group["objectId"]
                realobjids["objids2indids"][group["objectId"]] = object_id
                object_id += 1
    if agg_file.split("/")[-2] == "scene0217_00":
        object_ids = sorted(object_id2segs.keys())
        object_id2segs = {
            objectId: object_id2segs[objectId]
            for objectId in object_ids[: len(object_id2segs) // 2]
        }
    return object_id2segs, label2segs, realobjids


def read_seg_file(seg_file):
    seg2verts = {}
    with open(seg_file, "r") as json_data:
        data = json.load(json_data)
        num_verts = len(data["segIndices"])
        for vert, seg in enumerate(data["segIndices"]):
            if seg in seg2verts:
                seg2verts[seg].append(vert)
            else:
                seg2verts[seg] = [vert]
    return seg2verts, num_verts


def get_instance_ids(objectId2segs, seg2verts, sem_labels):
    object_id2label_id = {}
    # -1: points are not assigned to any objects ( objectId starts from 0)
    instance_ids = np.full(shape=len(sem_labels), fill_value=-1, dtype=np.int32)
    for objectId, segs in objectId2segs.items():
        for seg in segs:
            verts = seg2verts[seg]
            instance_ids[verts] = objectId
        if objectId not in object_id2label_id:
            object_id2label_id[objectId] = sem_labels[verts][0]

    return instance_ids, object_id2label_id


def get_instance_bboxes(xyz, instance_ids, object_id2label_id):
    num_instances = max(object_id2label_id.keys()) + 1
    instance_bboxes = np.zeros(
        shape=(num_instances, 8)
    )  # (cx, cy, cz, dx, dy, dz, ins_label, objectId)
    for objectId in object_id2label_id:
        ins_label = object_id2label_id[objectId]  # nyu40id
        obj_pc = xyz[instance_ids == objectId]  #
        if len(obj_pc) == 0:
            continue
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array(
            [
                (xmin + xmax) / 2,
                (ymin + ymax) / 2,
                (zmin + zmax) / 2,
                xmax - xmin,
                ymax - ymin,
                zmax - zmin,
                ins_label,
                objectId,
            ]
        )
        instance_bboxes[objectId, :] = bbox
    return instance_bboxes


def export(scene, cfg):
    mesh_file_path = os.path.join(cfg.raw_scan_path, scene, scene + "_vh_clean_2.ply")
    label_file_path = os.path.join(
        cfg.raw_scan_path, scene, scene + "_vh_clean_2.labels.ply"
    )
    agg_file_path = os.path.join(cfg.raw_scan_path, scene, scene + ".aggregation.json")
    seg_file_path = os.path.join(
        cfg.raw_scan_path, scene, scene + "_vh_clean_2.0.010000.segs.json"
    )

    # read mesh_file
    xyz, rgb, normal = read_mesh_file(mesh_file_path)
    num_verts = len(xyz)

    if os.path.exists(agg_file_path):
        # read label_file
        sem_labels = read_label_file(label_file_path)
        # read seg_file
        seg2verts, num = read_seg_file(seg_file_path)
        assert num_verts == num
        # read agg_file
        object_id2segs, label2segs, realobjids = read_agg_file(agg_file_path)
        # get instance labels
        instance_ids, object_id2label_id = get_instance_ids(
            object_id2segs, seg2verts, sem_labels
        )
        # get aligned instance bounding boxes
        aligned_instance_bboxes = get_instance_bboxes(
            xyz, instance_ids, object_id2label_id
        )
    else:
        # use zero as placeholders for the test scene
        # print("use placeholders")
        sem_labels = np.zeros(shape=num_verts, dtype=np.int32)  # 0: unannotated
        instance_ids = np.full(
            shape=num_verts, fill_value=-1, dtype=np.int32
        )  # -1: unannotated
        realobjids = {}
        aligned_instance_bboxes = np.zeros(shape=(1, 8), dtype=np.float32)
    sem_labels = remapper[sem_labels]
    return (
        xyz,
        rgb,
        normal,
        sem_labels,
        instance_ids,
        aligned_instance_bboxes,
        realobjids,
    )


def process_one_scan(scan, cfg, split):
    (
        xyz,
        rgb,
        normal,
        sem_labels,
        instance_ids,
        aligned_instance_bboxes,
        realobjids,
    ) = export(scan, cfg)

    # match the mesh2cap; not care wall, floor and ceiling for instances
    bbox_mask = np.logical_not(
        np.in1d(aligned_instance_bboxes[:, -2], IGNORE_CLASS_IDS)
    )
    aligned_instance_bboxes = aligned_instance_bboxes[bbox_mask, :]
    torch.save(
        {
            "xyz": xyz,
            "rgb": rgb,
            "normal": normal,
            "sem_labels": sem_labels,
            "instance_ids": instance_ids,
            "aligned_instance_bboxes": aligned_instance_bboxes,
            "realobjids": realobjids,
        },
        os.path.join(cfg.data.dataset_path, split, f"{scan}{cfg.data.file_suffix}"),
    )


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    cfg.project_root_path = cfg.project_root_path.rsplit("/", 2)[
        0
    ]  # hack the root path

    os.makedirs(os.path.join(cfg.data.dataset_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(cfg.data.dataset_path, "val"), exist_ok=True)
    os.makedirs(os.path.join(cfg.data.dataset_path, "test"), exist_ok=True)

    with open(cfg.data.metadata.train_list) as f:
        train_list = [line.strip() for line in f]

    with open(cfg.data.metadata.val_list) as f:
        val_list = [line.strip() for line in f]

    with open(cfg.data.metadata.test_list) as f:
        test_list = [line.strip() for line in f]

    print("==> Processing train split ...")
    process_map(
        partial(process_one_scan, cfg=cfg, split="train"), train_list, chunksize=1
    )
    print("==> Processing val split ...")
    process_map(partial(process_one_scan, cfg=cfg, split="val"), val_list, chunksize=1)
    print("==> Processing test split ...")
    process_map(
        partial(process_one_scan, cfg=cfg, split="test"), test_list, chunksize=1
    )


if __name__ == "__main__":
    main()
