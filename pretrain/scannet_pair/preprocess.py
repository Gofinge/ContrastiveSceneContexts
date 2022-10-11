import os
import argparse
import glob
import multiprocessing as mp
from reader import reader
from point_cloud_extractor import extractor
from compute_full_overlapping import compute_full_overlapping


frame_skip = 25


def parse_sens(sens_dir, output_dir):
    scene = os.path.basename(os.path.dirname(sens_dir))
    print(f"Parsing sens data{sens_dir}")
    reader(sens_dir, os.path.join(output_dir, scene), frame_skip,
           export_depth_images=True, export_poses=True, export_intrinsics=True)
    extractor(os.path.join(output_dir, scene), os.path.join(output_dir, scene, "pcd"))
    compute_full_overlapping(os.path.join(output_dir, scene, "pcd"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True, help='Path to the ScanNet dataset containing scene folders')
    parser.add_argument('--output_root', required=True, help='Output path where train/val folders will be located')
    config = parser.parse_args()
    sens_list = glob.glob(os.path.join(config.dataset_root, "scans/scene*/*.sens"))
    pool = mp.Pool(processes=mp.cpu_count())
    pool.starmap(parse_sens, [(sens) for sens in sens_list])
    pool.close()
    pool.join()
