import argparse
import json
import os
import sys

sys.path.append('../..')

import numpy as np
import open3d as o3d
from tqdm import tqdm

from util.bbox import write_cylinder_bbox
from util.pc import write_ply_rgb_face

SCANNET_COLOR_MAP = {
    0: (0.0, 0.0, 255.0),
    1: (0.0, 255.0, 0.0),
    2: (255.0, 0.0, 0.0),
}

def generate_bbox_ply(args, predicted_mask_list, labelIndexes, points, colors, indices, rgb_inst_ply):
    b_verts = []
    b_colors = []
    b_indices = []
    for index, predicted_mask in enumerate(predicted_mask_list):
        x_min = predicted_mask[0]
        y_min = predicted_mask[1]
        z_min = predicted_mask[2]
        x_max = predicted_mask[3]
        y_max = predicted_mask[4]
        z_max = predicted_mask[5]
        currbbox = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0, x_max - x_min, y_max - y_min,
                    z_max - z_min]

        semanticIndex = labelIndexes[index]
        chooseColor = SCANNET_COLOR_MAP[int(semanticIndex)]
        curr_verts, curr_colors, curr_indices = write_cylinder_bbox(np.array(currbbox), 0, None, color=chooseColor)
        curr_indices = np.array(curr_indices)
        curr_indices = curr_indices + len(b_verts)
        curr_indices = curr_indices.tolist()
        b_verts.extend(curr_verts)
        b_colors.extend(curr_colors)
        b_indices.extend(curr_indices)

    points = points.tolist()
    colors = colors.tolist()
    indices = indices.tolist()
    b_indices = np.array(b_indices)
    b_indices = b_indices + len(points)
    b_indices = b_indices.tolist()
    points.extend(b_verts)
    colors.extend(b_colors)
    indices.extend(b_indices)

    points = np.array(points)
    colors = np.array(colors)
    indices = np.array(indices)
    write_ply_rgb_face(points, colors, indices, rgb_inst_ply)
    return 0


def generate_single_ply(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # define position of necessary files
    ply_file = os.path.join(args.scans, args.scene_id, f'{args.scene_id}_vh_clean_2.ply')
    # ply_file_txt = os.path.join(args.scans, args.scene_id, f'{args.scene_id}.txt')
    
    # Load scene axis alignment matrix
    # lines = open(ply_file_txt).readlines()
    # axis_align_matrix = None
    # for line in lines:
    #     if 'axisAlignment' in line:
    #         axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    # axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))

    # get mesh
    if args.bbox_only:
        # define where to output the ply file
        rgb_inst_ply = os.path.join(args.output_dir, f'{args.question_id}_bbox.ply')
        
        points = np.empty((0, 3))
        colors = np.empty((0, 3))
        indices = np.empty((0, 3))
        labelIndexes = args.bboxes_labels
        predicted_mask_list = args.bboxes
    else:
        # define where to output the ply file
        rgb_inst_ply = os.path.join(args.output_dir, f'{args.question_id}.ply')
        
        scannet_data = o3d.io.read_triangle_mesh(ply_file)
        scannet_data.compute_vertex_normals()
        
        # R = scannet_data.get_rotation_matrix_from_xyz((0, 0, np.pi / 2))
        # scannet_data.rotate(R, center=(0, 0, 0))
        
        points = np.asarray(scannet_data.vertices)
        # pts = np.ones((points.shape[0], 4))
        # pts[:,0:3] = points[:,0:3]
        # pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
        # points = pts[:, 0:3]
        points -= points.mean(axis=0)
        colors = np.asarray(scannet_data.vertex_colors)
        indices = np.asarray(scannet_data.triangles)
        colors = colors * 255.0
        labelIndexes = args.bboxes_labels
        predicted_mask_list = args.bboxes

    generate_bbox_ply(args, predicted_mask_list, labelIndexes, points, colors, indices,
                        rgb_inst_ply)


def generate_pred_inst_ply(args):    
    # scans path
    if args.split == "test_wo_obj":
        args.scans = os.path.join('data/scannet/scans_test')
    else:
        args.scans = os.path.join('data/scannet/scans')
        
    args.scannet = os.path.join('data/scannet')
    
    # predictions path
    if args.split == "val":
        args.predictions_path = os.path.join('./output/', args.experiment_name, 'inference/predictions.json')
    else:
        args.predictions_path = os.path.join('./output/', args.experiment_name, f'prediction/predictions_{args.split}.json')
        
    # read predictions
    predictions = json.load(open(args.predictions_path))
    print(len(predictions), "predictions", args.split)
    
    # filter predictions
    predictions = [pred for pred in predictions if pred["question_id"] in args.filter_questions[args.split]]
    
    # scene ids
    scene_ids = [name for name in os.listdir('data/scannet/scans/')]

    for pred in tqdm(predictions):
        if args.split != "test_wo_obj":
            args.scene_id = sorted([scene_id for scene_id in scene_ids if pred["scene_id"][:-3] in scene_id])[0]
        else:
            args.scene_id = pred["scene_id"]
        args.question_id = pred["question_id"]
        if not args.no_bboxes:
            args.bboxes = [pred["bbox"]]
            args.bboxes_labels = [0]
            if "gt_bbox" in pred:
                args.bboxes.append(pred["gt_bbox"])
                args.bboxes_labels.append(1)
            if "scanqa_bbox" in pred:
                args.bboxes.append(pred["scanqa_bbox"])
                args.bboxes_labels.append(2)
        else:
            args.bboxes = []
            args.bboxes_labels = []
        generate_single_ply(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--split', type=str, default='val', choices=['test_w_obj', 'test_wo_obj', 'val'],
                        help='specify the split of data: val | test_w_obj | test_wo_obj')
    parser.add_argument('-o', '--output_dir', type=str, default='output_ply',
                        help='the directory of the output ply')
    parser.add_argument('-e', '--experiment_name', type=str, default='run_1',
                        help='name of the experiment to find the directory')
    parser.add_argument('--bbox_only', action="store_true", help="Output only bboxes")
    parser.add_argument('--no_bboxes', action="store_true", help="Output only scene")
    parser.set_defaults(bbox_only=False)
    parser.set_defaults(no_bboxes=False)
    args = parser.parse_args()

    args.filter_questions = {
        "val": [
            "val-scene0086-103"
        ],
        "test_w_obj": [
            "val-scene0583-13"
        ],
        "test_wo_obj": [
            "test-scene0724-31"
        ]
    }

    if not args.no_bboxes:
        args.output_dir = os.path.join(args.output_dir, args.split)
    else:
        args.output_dir = os.path.join(args.output_dir, "scenes")

    generate_pred_inst_ply(args)
