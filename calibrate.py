from __future__ import annotations
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from single_shot_calib.detector import camera_corners,lidar_lattice,recover_xyz
from single_shot_calib.config import load_config
from single_shot_calib.diagnostics import save as save_diagnostics
from single_shot_calib.io_utils import discover_direct_inputs,load_pair,save_json
from single_shot_calib.rosbag_input import extract_ros1,extract_ros2
from single_shot_calib.solver import solve
from single_shot_calib.spherical import project


def run_capture(name,image,xyz,intensity,cfg,output):
    start=time.perf_counter();board=cfg["checkerboard"]["inner_corners"];camera=cfg["camera"]
    K=np.asarray(camera["camera_matrix"],float);D=np.asarray(camera["distortion"],float);model=camera["model"]
    camera_grid=camera_corners(image,board,cfg["detection"]["camera_upscale_factors"])
    intensity_image,projected_uv,point_ids,_=project(xyz,intensity,cfg["spherical_projection"])
    lidar_grid,matched,candidates=lidar_lattice(intensity_image,board,cfg["detection"])
    lidar_xyz,lattice_indices=recover_xyz(lidar_grid,matched,projected_uv,point_ids,xyz,cfg["detection"]["point_recovery_radius_px"])
    if len(lidar_xyz)<cfg["detection"]["minimum_lidar_corners"]:raise RuntimeError(f"Only {len(lidar_xyz)} LiDAR grid corners recovered")
    rms,orientation,R,t,image_points,errors,initial_rms=solve(lidar_xyz,lattice_indices,camera_grid,board,K,D,model,cfg["optimization"])
    transform=np.eye(4);transform[:3,:3]=R;transform[:3,3]=t;capture_out=output/name;capture_out.mkdir(parents=True,exist_ok=True)
    quality_limit=float(cfg["quality"]["maximum_reprojection_rmse_px"])
    quality_status="pass" if rms<=quality_limit and rms<initial_rms else "review"
    save_diagnostics(capture_out,image,xyz,intensity_image,R,t,K,D,model,camera_grid,lidar_grid,matched,candidates,
                     {"quality_status":quality_status,"initial_rms":initial_rms,"rms":rms,"corners":len(lidar_xyz)})
    result={"method":"spherical intensity image + derivative lattice + E-PnP + LM","input_name":name,
      "transform_lidar_to_camera":transform.tolist(),"transform_camera_to_lidar":np.linalg.inv(transform).tolist(),
      "rotation_lidar_to_camera":R.tolist(),"translation_lidar_to_camera_m":t.tolist(),"lattice_orientation":orientation,
      "camera_checkerboard_corners":len(camera_grid),"matched_lidar_corners":len(lidar_xyz),
      "quality_status":quality_status,"quality_threshold_rmse_px":quality_limit,
      "epnp_initial_reprojection_rmse_px":initial_rms,"reprojection_rmse_px":rms,
      "mean_error_px":float(np.mean(errors)),"maximum_error_px":float(np.max(errors)),
      "per_corner_errors_px":errors.tolist(),"runtime_seconds":time.perf_counter()-start,
      "checkerboard_square_size_mm":cfg["checkerboard"].get("square_size_mm")}
    save_json(capture_out/"calibration.json",result);return result


def main():
    parser=argparse.ArgumentParser(description="Paper-method single-shot LiDAR-camera extrinsic calibration");parser.add_argument("--config",type=Path,default=Path("config.yaml"));args=parser.parse_args()
    root=Path(__file__).resolve().parent;cfg=load_config((root/args.config).resolve());output=(root/cfg["output_dir"]).resolve()
    if output.parent!=root:raise RuntimeError(f"Output directory must be directly inside the repository: {output}")
    output.mkdir(parents=True,exist_ok=True);mode=cfg["input"].get("mode","auto");results=[]
    if mode=="auto":
        inp=cfg["input"]
        if inp.get("bags"):
            first=root/inp["bags"][0];mode="ros1_bag" if first.suffix.lower()==".bag" else "ros2_bag"
        elif inp.get("pairs_dir"):mode="pairs_dir"
        elif inp.get("image_dir") and inp.get("pointcloud_dir"):mode="same_stem"
        else:mode="direct"
    if mode in {"direct","pairs_dir","same_stem"}:
        for name,image_path,cloud_path in discover_direct_inputs(root,cfg):
            image,xyz,intensity=load_pair(image_path,cloud_path);results.append(run_capture(name,image,xyz,intensity,cfg,output))
    elif mode in {"ros1_bag","ros2_bag"}:
        for bag in cfg["input"]["bags"]:
            path=root/bag;extract=extract_ros1 if mode=="ros1_bag" else extract_ros2
            image,xyz,intensity=extract(path,cfg["input"]["image_topic"],cfg["input"]["pointcloud_topic"],
                float(cfg["input"]["merge_window_sec"]),float(cfg["input"]["sync_tolerance_sec"]));results.append(run_capture(path.stem,image,xyz,intensity,cfg,output))
    else:raise ValueError(f"Unknown input mode: {mode}")
    save_json(output/"summary.json",{"captures":results})
    for r in results:print(f"{r['input_name']}: {r['reprojection_rmse_px']:.3f} px, {r['matched_lidar_corners']} LiDAR corners, {r['runtime_seconds']:.2f} s")
if __name__=="__main__":main()
