import math
from typing import Sequence, Tuple, Union

import numpy as np
import open3d as o3d
import torch
from open3d.j_visualizer import JVisualizer

# Mesh memory transformer
TensorOrSequence = Union[Sequence[torch.Tensor], torch.Tensor]
TensorOrNone = Union[torch.Tensor, None]


def get_batch_size(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.size(0)
    else:
        b_s = x[0].size(0)
    return b_s


def get_device(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.device
    else:
        b_s = x[0].device
    return b_s


def get_ious(box_a, box_b):
    """Computes IoU of axis aligned bboxes.
    Args:
        box_a, box_b: (N, xyzxyz)
    Returns:
        ious
    """
    result = np.zeros(box_a.shape[0])

    max_a = box_a[:, 3:]
    max_b = box_b[:, 3:]
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[:, 0:3]
    min_b = box_b[:, 0:3]
    max_min = np.array([min_a, min_b]).max(0)

    valid_indices = (min_max > max_min).all(axis=1)
    min_max = min_max[valid_indices]
    max_min = max_min[valid_indices]

    intersection = (min_max - max_min).prod(axis=1)
    vol_a = (box_a[valid_indices][:, 3:6] - box_a[valid_indices][:, :3]).prod(axis=1)
    vol_b = (box_b[valid_indices][:, 3:6] - box_b[valid_indices][:, :3]).prod(axis=1)
    union = vol_a + vol_b - intersection
    result[valid_indices] = 1.0 * intersection / union
    return result


def generate_mask(sequence):
    """
    :param sequence: (b, n, d) - b: batch, n: sequence length, d: embedding dimension
    :return: mask: (b, n) - boolean where it's 1 if n embedding's sum is 0, otherwise it's 0. b: batch, n: sequence length
    """
    return (torch.sum(sequence, 2) == 0).bool()


# https://pytorch.org/tutorials/beginner/translation_transformer.html
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def get_3d_box(center, box_size, heading_angle=None):
    """box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
    output (8,3) array for 3D box cornders
    Similar to util/compute_orientation_3d
    """
    if heading_angle is None:
        R = np.eye(3)
    else:
        R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)  # (8, 3)
    return corners_3d


def get_3d_box_edges(corners):
    """
    Args:
        corners: (8,3) array for 3D box cornders returned by get_3d_box
    Output:
        edges: a list of size 12, where each entry is a pair of end points representing an edge
    """
    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
        (corners[4], corners[5]),
        (corners[5], corners[6]),
        (corners[6], corners[7]),
        (corners[7], corners[4]),
        (corners[0], corners[4]),
        (corners[1], corners[5]),
        (corners[2], corners[6]),
        (corners[3], corners[7]),
    ]
    return edges


def write_cylinder_bbox(bbox, mode, out_filename=None, color=None):
    """
    bbox: (cx, cy, cz, lx, ly, lz, r), center and length in three axis, the last is the rotation
        or (cx, cy, cz, lx, ly, lz)

    out_filename: string
    """

    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])

        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0, 0] = 1 + t * (x * x - 1)
            rot[0, 1] = z * s + t * x * y
            rot[0, 2] = -y * s + t * x * z
            rot[1, 0] = -z * s + t * x * y
            rot[1, 1] = 1 + t * (y * y - 1)
            rot[1, 2] = x * s + t * y * z
            rot[2, 0] = y * s + t * x * z
            rot[2, 1] = -x * s + t * y * z
            rot[2, 2] = 1 + t * (z * z - 1)
            return rot

        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks + 1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array(
                    [
                        radius * math.cos(theta),
                        radius * math.sin(theta),
                        height * i / stacks,
                    ]
                )
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append(
                    np.array(
                        [(i + 1) * slices + i2, i * slices + i2, i * slices + i2p1],
                        dtype=np.uint32,
                    )
                )
                indices.append(
                    np.array(
                        [
                            (i + 1) * slices + i2,
                            i * slices + i2p1,
                            (i + 1) * slices + i2p1,
                        ],
                        dtype=np.uint32,
                    )
                )
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if math.fabs(dotx) != 1.0:
                    axis = np.array([1, 0, 0]) - dotx * va
                else:
                    axis = np.array([0, 1, 0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3, 3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]

        return verts, indices

    radius = 0.03
    offset = [0, 0, 0]
    verts = []
    indices = []
    colors = []
    if bbox.size == 24:  # corners 8*3
        corners = bbox
    elif bbox.size == 6:
        corners = get_3d_box(bbox[:3], bbox[3:6])
    else:
        corners = get_3d_box(bbox[:3], bbox[3:6], bbox[6])

    palette = {0: [0, 255, 0], 1: [0, 0, 255]}  # gt  # pred
    chosen_color = palette[mode]
    if color is not None:
        chosen_color = color
    edges = get_3d_box_edges(corners)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c for c in chosen_color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)

    return verts, colors, indices


def get_bbox_points(bbox, color):
    x_min = bbox[0]
    y_min = bbox[1]
    z_min = bbox[2]
    x_max = bbox[3]
    y_max = bbox[4]
    z_max = bbox[5]
    currbbox = [
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        (z_min + z_max) / 2.0,
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
    ]
    curr_verts, curr_colors, curr_indices = write_cylinder_bbox(
        np.array(currbbox), 0, None, color=color
    )
    return curr_verts, curr_colors


def plot(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    visualizer = JVisualizer()
    visualizer.add_geometry(pcd)
    visualizer.show()


def show_prediction(sample, dataset, plot_scene=True, prediction=False):
    scene_id = sample["scene_id"]
    scene = dataset.scenes[scene_id]
    points = scene["xyz"]
    colors = scene["rgb"] + 1

    if not prediction:
        # ground truth
        gt_bbox = sample["gt_bbox"]
        gt_bbox_full = get_bbox_points(np.array(gt_bbox), ((0.0, 255.0, 0.0)))
        points = np.concatenate((points, np.array(gt_bbox_full[0])), axis=0)
        colors = np.concatenate((colors, np.array(gt_bbox_full[1])), axis=0)

    # sample data
    pred_bbox_sample = sample["bbox"]
    pred_bbox_sample_full = get_bbox_points(pred_bbox_sample, ((255.0, 0.0, 0.0)))
    points = np.concatenate((points, np.array(pred_bbox_sample_full[0])), axis=0)
    colors = np.concatenate((colors, np.array(pred_bbox_sample_full[1])), axis=0)

    G_LABEL_NAMES = [
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refrigerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "others",
        "background",
    ]

    print(scene_id)
    print(sample["question_id"])
    print()

    print(sample["question"])
    if not prediction:
        print("GT answers: ", sample["gt_answers"])
    print("PRED answers: ", sample["answer_top10"])
    if "scanqa_answer" in sample:
        print("SCANQA answers: ", sample["scanqa_answer"])
    print()

    if not prediction:
        print("GT object classes: ", sample["gt_object_classes"])
    print("PRED object class: ", sample["pred_object_class"])
    print()

    if not prediction:
        print("PRED bbox iou: ", sample["iou"])

    if plot_scene:
        plot(points, colors)
