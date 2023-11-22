import argparse
import json
import os
import random
import sys

sys.path.append('../..')
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from util.bbox import write_cylinder_bbox

SCANNET_COLOR_MAP = {
    0: (255.0, 0.0, 0.0),
    1: (0.0, 255.0, 0.0),
    2: (0.0, 0.0, 255.0),
}

def write_ply_rgb_face(points, colors, faces, filename, text=True):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as ply file """
    colors = colors.astype(int)
    points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
    faces = [((faces[i,0], faces[i,1], faces[i,2]),) for i in range(faces.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    face = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])
    ele1 = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    ele2 = PlyElement.describe(face, 'face', comments=['faces'])
    PlyData([ele1, ele2], text=text).write(filename)


def get_bbox(predicted_mask, points):
    x_min = None
    y_min = None
    z_min = None
    x_max = None
    y_max = None
    z_max = None
    for vertexIndex, xyz in enumerate(points):
        if predicted_mask[vertexIndex] == True:
            if x_min is None or xyz[0] < x_min:
                x_min = xyz[0]
            if y_min is None or xyz[1] < y_min:
                y_min = xyz[1]
            if z_min is None or xyz[2] < z_min:
                z_min = xyz[2]
            if x_max is None or xyz[0] > x_max:
                x_max = xyz[0]
            if y_max is None or xyz[1] > y_max:
                y_max = xyz[1]
            if z_max is None or xyz[2] > z_max:
                z_max = xyz[2]
    return x_min, x_max, y_min, y_max, z_min, z_max


def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return [r, g, b]


def get_random_rgb_colors(num):
    rgb_colors = [get_random_color() for _ in range(num)]
    return rgb_colors


def generate_colored_ply(args, predicted_mask_list, labelIndexes, points, colors, indices,
                         rgb_inst_ply):
    if args.mode == "semantic":
        for index, predicted_mask in enumerate(predicted_mask_list):
            semanticIndex = labelIndexes[index]
            # confidence = confidenceScores[index]
            for vertexIndex, color in enumerate(colors):
                if predicted_mask[vertexIndex] == True:
                    colors[vertexIndex] = SCANNET_COLOR_MAP[int(semanticIndex)]
    elif args.mode == "instance":
        color_list = get_random_rgb_colors(len(labelIndexes))
        random.shuffle(color_list)
        for index, predicted_mask in enumerate(predicted_mask_list):
            for vertexIndex, color in enumerate(colors):
                if predicted_mask[vertexIndex] == True:
                    colors[vertexIndex] = color_list[index]
    write_ply_rgb_face(points, colors, indices, rgb_inst_ply)
    return 0


def generate_bbox_ply(args, predicted_mask_list, labelIndexes, points, colors, indices, rgb_inst_ply):
    b_verts = []
    b_colors = []
    b_indices = []
    for index, predicted_mask in enumerate(predicted_mask_list):
        # x_min, x_max, y_min, y_max, z_min, z_max = get_bbox(predicted_mask, points)
        predicted_mask = predicted_mask[0] + predicted_mask[-2]
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
    ply_file = os.path.join(args.scans, f'{args.scene_id}_aligned_vert.npy')
    if args.split == "test_wo_obj":
        ply_file_indices = os.path.join(args.scans, '../scans_test', args.scene_id, f'{args.scene_id}_vh_clean_2.ply')
        ply_file_txt = os.path.join(args.scans, '../scans_test', args.scene_id, f'{args.scene_id}.txt')
    else:
        ply_file_indices = os.path.join(args.scans, '../scans', args.scene_id, f'{args.scene_id}_vh_clean_2.ply')
        ply_file_txt = os.path.join(args.scans, '../scans', args.scene_id, f'{args.scene_id}.txt')
    # pth_file = os.path.join(args.scannet, args.mode, f'{args.scene_id}.pth')

    # define where to output the ply file
    rgb_inst_ply = os.path.join(args.output_dir, f'{args.question_id}.ply')


    # mesh_vertices = np.load(ply_file)
    # points = mesh_vertices[:,0:3]
    # colors = mesh_vertices[:,3:6]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.estimate_normals()
    # scannet_data = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.15)
    # scannet_data.compute_vertex_normals()
    
    # Load scene axis alignment matrix
    if args.split != "test_wo_obj":
        lines = open(ply_file_txt).readlines()
        axis_align_matrix = None
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))

    # get mesh
    scannet_data = o3d.io.read_triangle_mesh(ply_file_indices)
    scannet_data.compute_vertex_normals()
    points = np.asarray(scannet_data.vertices)
    pts = np.ones((points.shape[0], 4))
    pts[:,0:3] = points[:,0:3]
    if args.split != "test_wo_obj":
        pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    points = pts[:, 0:3]
    # points -= points.mean(axis=0)
    colors = np.asarray(scannet_data.vertex_colors)
    indices = np.asarray(scannet_data.triangles)
    colors = colors * 255.0

    # with open(pred_sem_file) as file:
    #     lines = file.readlines()
    #     lines = [line.rstrip() for line in lines]
    lines = []

    instanceFileNames = []
    labelIndexes = args.bboxes_labels
    confidenceScores = []
    predicted_mask_list = args.bboxes
    for i in lines:
        splitedLine = i.split()
        instanceFileNames.append(os.path.join(args.predict_dir, splitedLine[0]))
        labelIndexes.append(splitedLine[1])
        confidenceScores.append(splitedLine[2])

    for instanceFileName in instanceFileNames:
        predicted_mask_list.append(np.loadtxt(instanceFileName, dtype=bool))

    generate_bbox_ply(args, predicted_mask_list, labelIndexes, points, colors, indices,
                        rgb_inst_ply)


def generate_pred_inst_ply(args):    
    # scans path
    if args.split == "test_wo_obj":
        args.scans = os.path.join('data/scannet/scannet_data')
    else:
        args.scans = os.path.join('data/scannet/scannet_data')
        
    args.scannet = os.path.join('data/scannet')
    
    # predictions path
    args.predictions_path = os.path.join('./outputs/', args.experiment_name, f'pred.{args.split}.json')
        
    # read predictions
    predictions = json.load(open(args.predictions_path))
    print(len(predictions), "predictions", args.split)
    
    # filter predictions
    predictions = [pred for pred in predictions if pred["question_id"] in args.filter_questions[args.split]]

    for pred in tqdm(predictions):
        args.scene_id = pred["scene_id"]
        args.question_id = pred["question_id"]
        args.bboxes = [pred["bbox"]]
        args.bboxes_labels = [0]
        if "gt_bbox" in pred:
            args.bboxes.append(pred["gt_bbox"])
            args.bboxes_labels.append(1)
        if "scanqa_bbox" in pred:
            args.bboxes.append(pred["scanqa_bbox"])
            args.bboxes_labels.append(2)
        generate_single_ply(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--split', type=str, default='val', choices=['test_w_obj', 'test_wo_obj', 'val'],
                        help='specify the split of data: val | test_w_obj | test_wo_obj')
    parser.add_argument('-o', '--output_dir', type=str, default='output_ply',
                        help='the directory of the output ply')
    parser.add_argument('-e', '--experiment_name', type=str, default='XYZ_MULTIVIEW_NORMAL',
                        help='name of the experiment to find the directory')
    args = parser.parse_args()
    
    args.filter_questions = {
        "val": [
            "val-scene0086-103"    
        ],
        "test_w_obj": [],
        "test_wo_obj": [
            "test-scene0770-80",
            "test-scene0724-31",
            "test-scene0793-25",
            "test-scene0755-97",
            "test-scene0762-7",
            "test-scene0759-59",
            "test-scene0766-21",
            "test-scene0721-128",
            "test-scene0746-2"
        ]
    }

    args.output_dir = os.path.join(args.output_dir, args.split)

    generate_pred_inst_ply(args)
