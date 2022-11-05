import os
import sys
import plyfile
import json
import time
import torch
import argparse
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec ** 2, axis=1, keepdims=True)) + 1.0e-8
    nf = vec / length
    area = length * 0.5
    return nf, area


def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area

    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]

    length = np.sqrt(np.sum(nv ** 2, axis=1, keepdims=True)) + 1.0e-8
    nv = nv / length
    return nv

def get_raw2scannet_label_map():
    lines = [line.rstrip() for line in open('scannetv2-labels.combined.tsv')]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        elements = lines[i].split('\t')
        # raw_name = elements[0]
        # nyu40_name = elements[6]
        raw_name = elements[1]
        nyu40_id = elements[4]
        nyu40_name = elements[7]
        raw2scannet[raw_name] = nyu40_id
    return raw2scannet
g_raw2scannet = get_raw2scannet_label_map()
RAW2SCANNET = g_raw2scannet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/canis/Datasets/ScanNet/public/v2/scans/')
    parser.add_argument('--output', default='./output')
    opt = parser.parse_args()
    return opt

def main(scene_name, input, output):
    print(scene_name)
    # Over-segmented segments: maps from segment to vertex/point IDs
    segid_to_pointid = {}
    segfile = os.path.join(input, scene_name, '%s_vh_clean_2.0.010000.segs.json' % (scene_name))
    with open(segfile) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    # Raw points in XYZRGBA
    ply_filename = os.path.join(input, scene_name, '%s_vh_clean_2.ply' % (scene_name))
    f = plyfile.PlyData().read(ply_filename)
    points = np.array([list(x) for x in f.elements[0]])
    faces = np.stack(f['face'].data['vertex_indices'], axis=0)

    # Instances over-segmented segment IDs: annotation on segments
    instance_segids = []
    labels = []
    annotation_filename = os.path.join(input, scene_name, '%s.aggregation.json' % (scene_name))
    with open(annotation_filename) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            instance_segids.append(x['segments'])
            labels.append(x['label'])

    # Each instance's points
    instance_labels = np.zeros(points.shape[0])
    semantic_labels = np.zeros(points.shape[0])
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        pointids = np.array(pointids)
        instance_labels[pointids] = i + 1
        semantic_labels[pointids] = RAW2SCANNET[labels[i]]

    colors = points[:, 3:6]
    coords = points[:, 0:3]  # XYZ+RGB+NORMAL
    normals = vertex_normal(coords, faces)
    torch.save((coords, colors, semantic_labels, instance_labels, normals),
               os.path.join(output, scene_name + '.pth'))


if __name__=='__main__':
    config = parse_args()
    os.makedirs(config.output, exist_ok=True)
    data_list = os.listdir(config.input)

    # Preprocess data.
    pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    # pool = ProcessPoolExecutor(max_workers=1)
    print('Processing scenes...')
    _ = list(pool.map(main, data_list, repeat(config.input), repeat(config.output)))
