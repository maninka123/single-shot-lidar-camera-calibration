from __future__ import annotations
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from single_shot_calib.detector import camera_corners,lidar_lattice,recover_xyz
from single_shot_calib.config import load_config
from single_shot_calib.diagnostics import save as save_diagnostics
from single_shot_calib.io_utils import load_capture,save_json
from single_shot_calib.rosbag_input import extract_robust,extract_ros1,extract_ros2
from single_shot_calib.solver import solve
from single_shot_calib.spherical import project


def run_capture_robust(image,xyz,intensity,cfg,output,frame=None,capture_info=None):
    """Same single-shot core as run_capture, with the robust refinements (README: Robust mode)."""
    from single_shot_calib import robust
    from single_shot_calib.diagnostics import save_robust
    from single_shot_calib.solver import solve_robust
    start=time.perf_counter();board=cfg["checkerboard"]["inner_corners"];camera=cfg["camera"];rb=cfg["robust"]
    K=np.asarray(camera["camera_matrix"],float);D=np.asarray(camera["distortion"],float);model=camera["model"]
    camera_grid,cam_info=robust.camera_corners(image,board,cfg["detection"]["camera_upscale_factors"],rb["clahe_fallback"])
    square_mm=cfg["checkerboard"].get("square_size_mm");square_m=float(square_mm)/1000 if square_mm else None
    det=robust.detect_board(xyz,intensity,board,cfg,square_m,frame)
    notes=robust.gates(det,square_m,cfg)
    best,solutions,solver_notes=solve_robust(det["corners_xyz"],det["lattice_indices"],camera_grid,board,K,D,model,
                                             cfg["optimization"],rb,det["fit"]["normal"])
    notes+=solver_notes
    if best["rotation_std_deg"]>rb["maximum_rotation_std_deg"] or best["translation_std_mm"]>rb["maximum_translation_std_mm"]:
        notes.append(f"weak geometry: uncertainty {best['rotation_std_deg']:.1f} deg / {best['translation_std_mm']:.0f} mm "
                     "(board small or far; use a closer/larger board or several captures)")
    R,t,rms=best["R"],best["t"],best["rms"]
    transform=np.eye(4);transform[:3,:3]=R;transform[:3,3]=t;output.mkdir(parents=True,exist_ok=True)
    quality_limit=float(cfg["quality"]["maximum_reprojection_rmse_px"])
    quality_status="pass" if rms<=quality_limit and rms<=best["initial_rms"]+1e-9 and not notes else "review"
    save_robust(output,image,xyz,det,cfg["spherical_projection"],R,t,K,D,model,best["image_points"],det["corners_xyz"],
                {"quality_status":quality_status,"initial_rms":best["initial_rms"],"rms":rms,"corners":len(det["corners_xyz"]),
                 "board_range_m":det["board_range_m"],"square_mm":det["square_size_m"]*1000,"notes":notes})
    result={"method":"robust single-shot: paper core (spherical intensity image + lattice + E-PnP + LM) + robust refinements",
      "transform_lidar_to_camera":transform.tolist(),"transform_camera_to_lidar":np.linalg.inv(transform).tolist(),
      "rotation_lidar_to_camera":R.tolist(),"translation_lidar_to_camera_m":t.tolist(),"lattice_orientation":best["variant"],
      "camera_checkerboard_corners":len(camera_grid),"matched_lidar_corners":len(det["corners_xyz"]),
      "lidar_corner_source":"board-plane pattern fit (initialised by the paper lattice / range segments)",
      "quality_status":quality_status,"quality_threshold_rmse_px":quality_limit,"review_reasons":notes,
      "epnp_initial_reprojection_rmse_px":best["initial_rms"],"reprojection_rmse_px":rms,
      "mean_error_px":float(np.mean(best["errors"])),"maximum_error_px":float(np.max(best["errors"])),
      "per_corner_errors_px":best["errors"].tolist(),
      "uncertainty":{"rotation_std_deg":best["rotation_std_deg"],"translation_std_mm":best["translation_std_mm"]},
      "alternative_solutions":[{"orientation":s["variant"],"rms_px":s["rms"],"faces_both_sensors":s["facing"],
                                "angle_to_selected_deg":float(np.degrees(np.arccos(np.clip((np.trace(s["R"]@R.T)-1)/2,-1,1))))}
                               for s in solutions if s is not best],
      "camera":{"corner_detection":cam_info,**(capture_info or {})},
      "lidar_board":{"range_m":det["board_range_m"],"points":det["board_points"],"square_size_mm_used":det["square_size_m"]*1000,
                     "square_size_mm_free_fit":det["free_square_size_m"]*1000,"cell_contrast":det["contrast"],
                     "cell_coverage":det["cell_coverage"],"drift_mm":det["board_drift_mm"],"candidates":det["candidates"],
                     "paper_lattice_found":det["lattice"] is not None},
      "runtime_seconds":time.perf_counter()-start,"checkerboard_square_size_mm":square_mm}
    save_json(output/"calibration.json",result);return result


def run_capture(image,xyz,intensity,cfg,output):
    if cfg.get("robust",{}).get("enabled"):return run_capture_robust(image,xyz,intensity,cfg,output)
    start=time.perf_counter();board=cfg["checkerboard"]["inner_corners"];camera=cfg["camera"]
    K=np.asarray(camera["camera_matrix"],float);D=np.asarray(camera["distortion"],float);model=camera["model"]
    camera_grid=camera_corners(image,board,cfg["detection"]["camera_upscale_factors"])
    intensity_image,projected_uv,point_ids,_=project(xyz,intensity,cfg["spherical_projection"])
    lidar_grid,matched,candidates=lidar_lattice(intensity_image,board,cfg["detection"])
    lidar_xyz,lattice_indices=recover_xyz(lidar_grid,matched,projected_uv,point_ids,xyz,cfg["detection"]["point_recovery_radius_px"])
    if len(lidar_xyz)<cfg["detection"]["minimum_lidar_corners"]:raise RuntimeError(f"Only {len(lidar_xyz)} LiDAR grid corners recovered")
    rms,orientation,R,t,image_points,errors,initial_rms=solve(lidar_xyz,lattice_indices,camera_grid,board,K,D,model,cfg["optimization"])
    transform=np.eye(4);transform[:3,:3]=R;transform[:3,3]=t;output.mkdir(parents=True,exist_ok=True)
    quality_limit=float(cfg["quality"]["maximum_reprojection_rmse_px"])
    quality_status="pass" if rms<=quality_limit and rms<initial_rms else "review"
    save_diagnostics(output,image,xyz,intensity_image,R,t,K,D,model,camera_grid,lidar_grid,matched,candidates,
                     {"quality_status":quality_status,"initial_rms":initial_rms,"rms":rms,"corners":len(lidar_xyz)})
    result={"method":"spherical intensity image + derivative lattice + E-PnP + LM",
      "transform_lidar_to_camera":transform.tolist(),"transform_camera_to_lidar":np.linalg.inv(transform).tolist(),
      "rotation_lidar_to_camera":R.tolist(),"translation_lidar_to_camera_m":t.tolist(),"lattice_orientation":orientation,
      "camera_checkerboard_corners":len(camera_grid),"matched_lidar_corners":len(lidar_xyz),
      "quality_status":quality_status,"quality_threshold_rmse_px":quality_limit,
      "epnp_initial_reprojection_rmse_px":initial_rms,"reprojection_rmse_px":rms,
      "mean_error_px":float(np.mean(errors)),"maximum_error_px":float(np.max(errors)),
      "per_corner_errors_px":errors.tolist(),"runtime_seconds":time.perf_counter()-start,
      "checkerboard_square_size_mm":cfg["checkerboard"].get("square_size_mm")}
    save_json(output/"calibration.json",result);return result


def main():
    parser=argparse.ArgumentParser(description="Paper-method single-shot LiDAR-camera extrinsic calibration");parser.add_argument("--config",type=Path,default=Path("config.yaml"));args=parser.parse_args()
    root=Path(__file__).resolve().parent;cfg=load_config((root/args.config).resolve());output=(root/cfg["output_dir"]).resolve()
    if output.parent!=root:raise RuntimeError(f"Output directory must be directly inside the repository: {output}")
    output.mkdir(parents=True,exist_ok=True);inp=cfg["input"]
    if inp.get("bag") and cfg["robust"]["enabled"]:
        image,xyz,intensity,frame,info=extract_robust((root/inp["bag"]).resolve(),inp["image_topic"],inp["pointcloud_topic"],
                                                     cfg["robust"],float(inp["merge_window_sec"]))
        result=run_capture_robust(image,xyz,intensity,cfg,output,frame,info)
        print(f"{result['quality_status'].upper()}: {result['reprojection_rmse_px']:.3f} px, "
              f"{result['matched_lidar_corners']} LiDAR corners, {result['runtime_seconds']:.2f} s")
        for note in result["review_reasons"]:print(f"  review: {note}")
        return
    if inp.get("bag"):
        path=(root/inp["bag"]).resolve();extract=extract_ros1 if path.suffix.lower()==".bag" else extract_ros2
        image,xyz,intensity=extract(path,inp["image_topic"],inp["pointcloud_topic"],
            float(inp["merge_window_sec"]),float(inp["sync_tolerance_sec"]))
    else:
        image,xyz,intensity=load_capture((root/inp["image"]).resolve(),(root/inp["pointcloud"]).resolve())
    result=run_capture(image,xyz,intensity,cfg,output)
    print(f"{result['quality_status'].upper()}: {result['reprojection_rmse_px']:.3f} px, "
          f"{result['matched_lidar_corners']} LiDAR corners, {result['runtime_seconds']:.2f} s")
if __name__=="__main__":main()
