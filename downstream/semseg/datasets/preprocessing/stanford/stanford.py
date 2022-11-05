# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import glob
import numpy as np
import os
import torch
import errno

from tqdm import tqdm
import os
import numpy as np
from numpy.linalg import matrix_rank, inv
from plyfile import PlyData, PlyElement
import pandas as pd

COLOR_MAP_RGB = (
    (241, 255, 82),
    (102, 168, 226),
    (0, 255, 0),
    (113, 143, 65),
    (89, 173, 163),
    (254, 158, 137),
    (190, 123, 75),
    (100, 22, 116),
    (0, 18, 141),
    (84, 84, 84),
    (85, 116, 127),
    (255, 31, 33),
    (228, 228, 228),
    (0, 255, 0),
    (70, 145, 150),
    (237, 239, 94),
)
IGNORE_COLOR = (0, 0, 0)


def read_plyfile(filepath):
  """Read ply file and return it as numpy array. Returns None if emtpy."""
  with open(filepath, 'rb') as f:
    plydata = PlyData.read(f)
  if plydata.elements:
    return pd.DataFrame(plydata.elements[0].data).values


def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True):
  """Save an RGB point cloud as a PLY file.

  Args:
    points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
        the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
  """
  assert points_3d.ndim == 2
  if with_label:
    assert points_3d.shape[1] == 7
    python_types = (float, float, float, int, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1'), ('label', 'u1')]
  else:
    if points_3d.shape[1] == 3:
      gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
      points_3d = np.hstack((points_3d, gray_concat))
    assert points_3d.shape[1] == 6
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
  if binary is True:
    # Format into NumPy structured array
    vertices = []
    for row_idx in range(points_3d.shape[0]):
      cur_point = points_3d[row_idx]
      vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(vertices_array, 'vertex')

    # Write
    PlyData([el]).write(filename)
  else:
    # PlyData([el], text=True).write(filename)
    with open(filename, 'w') as f:
      f.write('ply\n'
              'format ascii 1.0\n'
              'element vertex %d\n'
              'property float x\n'
              'property float y\n'
              'property float z\n'
              'property uchar red\n'
              'property uchar green\n'
              'property uchar blue\n'
              'property uchar alpha\n'
              'end_header\n' % points_3d.shape[0])
      for row_idx in range(points_3d.shape[0]):
        X, Y, Z, R, G, B = points_3d[row_idx]
        f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
  if verbose is True:
    print('Saved point cloud to: %s' % filename)



def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

import MinkowskiEngine as ME

STANFORD_3D_IN_PATH = '/home/gofinge/Documents/datasets/Stanford3dDataset_v1.2'
STANFORD_3D_OUT_PATH = '/home/gofinge/Documents/datasets/processed/s3dis_csc'

STANFORD_3D_TO_SEGCLOUD_LABEL = {
    4: 0,
    8: 1,
    12: 2,
    1: 3,
    6: 4,
    13: 5,
    7: 6,
    5: 7,
    11: 8,
    3: 9,
    9: 10,
    2: 11,
    0: 12,
}


class Stanford3DDatasetConverter:

  CLASSES = [
      'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor', 'sofa',
      'stairs', 'table', 'wall', 'window'
  ]
  TRAIN_TEXT = 'train'
  VAL_TEXT = 'val'
  TEST_TEXT = 'test'

  @classmethod
  def read_txt(cls, txtfile):
    # Read txt file and parse its content.
    with open(txtfile) as f:
      pointcloud = []
      for l in f:
        try:
          pointcloud += [[float(li) for li in l.split()]]
        except Exception as e:
          print(e, txtfile)
          continue
      # pointcloud = [l.split() for l in f]

    # Load point cloud to named numpy array.
    pointcloud = np.array(pointcloud).astype(np.float32)
    assert pointcloud.shape[1] == 6
    xyz = pointcloud[:, :3].astype(np.float32)
    rgb = pointcloud[:, 3:].astype(np.uint8)
    return xyz, rgb

  @classmethod
  def convert_to_ply(cls, root_path, out_path, save_pth=False):
    """Convert Stanford3DDataset to PLY format that is compatible with
    Synthia dataset. Assumes file structure as given by the dataset.
    Outputs the processed PLY files to `STANFORD_3D_OUT_PATH`.
    """

    txtfiles = glob.glob(os.path.join(root_path, '*/*/*.txt'))
    for txtfile in tqdm(txtfiles):
      file_sp = os.path.normpath(txtfile).split(os.path.sep)
      target_path = os.path.join(out_path, file_sp[-3])
      out_file = os.path.join(target_path, file_sp[-2] + '.ply')
      if save_pth:
        out_file = os.path.join(target_path, file_sp[-2] + '.pth')


      if os.path.exists(out_file):
        print(out_file, ' exists')
        continue

      annotation, _ = os.path.split(txtfile)
      subclouds = glob.glob(os.path.join(annotation, 'Annotations/*.txt'))
      coords, feats, labels, instances = [], [], [], []
      for inst, subcloud in enumerate(subclouds):
        # Read ply file and parse its rgb values.
        xyz, rgb = cls.read_txt(subcloud)
        _, annotation_subfile = os.path.split(subcloud)
        clsidx = cls.CLASSES.index(annotation_subfile.split('_')[0])

        coords.append(xyz)
        feats.append(rgb)
        labels.append(np.ones((len(xyz), 1), dtype=np.int32) * clsidx)
        instances.append(np.ones((len(xyz), 1), dtype=np.int32) * inst)


      if len(coords) == 0:
        print(txtfile, ' has 0 files.')
      else:
        # Concat
        coords = np.concatenate(coords, 0)
        feats = np.concatenate(feats, 0)
        labels = np.concatenate(labels, 0)
        instances = np.concatenate(instances, 0)

        inds, collabels = ME.utils.sparse_quantize(
            coords,
            feats,
            labels,
            return_index=True,
            ignore_label=255,
            quantization_size=0.01  # 1cm
        )
        pointcloud = np.concatenate((coords[inds], feats[inds], collabels[:, None]), axis=1)
        if save_pth:
          pointcloud = np.concatenate((coords[inds], feats[inds], collabels[:, None], instances[inds]), axis=1)

        # Write ply file.
        mkdir_p(target_path)
        if save_pth:
          torch.save(pointcloud, out_file)
          continue
        save_point_cloud(pointcloud, out_file, with_label=True, verbose=False)


def generate_splits(stanford_out_path, suffix='ply'):
  """Takes preprocessed out path and generate txt files"""
  split_path = './splits/stanford'
  mkdir_p(split_path)
  for i in range(1, 7):
    curr_path = os.path.join(stanford_out_path, f'Area_{i}')
    files = glob.glob(os.path.join(curr_path, '*.{}'.format(suffix)))
    files = [os.path.relpath(full_path, stanford_out_path) for full_path in files]
    out_txt = os.path.join(split_path, f'area{i}.txt')
    with open(out_txt, 'w') as f:
      f.write('\n'.join(files))


if __name__ == '__main__':
  Stanford3DDatasetConverter.convert_to_ply(STANFORD_3D_IN_PATH, STANFORD_3D_OUT_PATH, save_pth=True)
  generate_splits(STANFORD_3D_OUT_PATH, 'pth')
