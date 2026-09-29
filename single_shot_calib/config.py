from copy import deepcopy
from pathlib import Path
import yaml

DEFAULTS={
 "input":{"image":"data/image.png","pointcloud":"data/cloud.pcd","bag":None,"image_topic":"/camera/image_raw","pointcloud_topic":"/livox/lidar","sync_tolerance_sec":0.05,"merge_window_sec":1.0},
 "checkerboard":{"square_size_mm":None},
 "spherical_projection":{"width":600,"height":600,"horizontal_fov_deg":90.0,"vertical_fov_deg":90.0,"forward_axis":[1,0,0],"right_axis":[0,-1,0],"down_axis":[0,0,-1],"minimum_range_m":0.35,"maximum_range_m":20.0},
 "detection":{"camera_upscale_factors":[1,2,3,4,5,6],"maximum_corner_candidates":1800,"corner_quality":0.003,"minimum_corner_spacing_px":3.0,"lattice_trials":5000,"lattice_tolerance_fraction":0.38,"minimum_lidar_corners":18,"point_recovery_radius_px":2.5,"random_seed":17},
 "optimization":{"maximum_iterations":300,"ftol":1e-10,"xtol":1e-10,"gtol":1e-10},
 "quality":{"maximum_reprojection_rmse_px":3.0},"output_dir":"outputs",
 # Reliability updates (README "Updates").  enabled: false = original published pipeline.
 "robust":{"enabled":True,
  "merge_full_capture":True,"camera_median":True,"maximum_frame_deviation":6.0,
  "clahe_fallback":True,"fill_holes":True,
  "board_roi":True,"depth_jump_m":0.08,"depth_jump_fraction":0.03,"board_area_m2":[0.04,0.8],"minimum_region_px":120,
  "bridge_cut_px":5,"roi_grow_px":9,"maximum_regions":8,"crop_margin_px":15,
  "minimum_points":300,"range_gate_m":0.35,"cluster_gap_m":0.04,"plane_tolerance_m":0.03,"board_thickness_m":0.10,
  "corner_square":"black","raster_m":0.004,"square_search_m":[0.02,0.12],"square_step_m":0.004,"angle_step_deg":3,
  "minimum_contrast":1.4,"minimum_cell_coverage":0.95,"square_size_tolerance":0.05,"maximum_board_drift_mm":10.0,
  "reject_mirrored":True,"approx_rotation_lidar_to_camera":None,"approx_rotation_tolerance_deg":45.0,
  "loss":"huber","loss_scale_px":1.0,"uncertainty_noise_floor_px":0.25,
  "maximum_rotation_std_deg":5.0,"maximum_translation_std_mm":300.0}}

def _merge(base,custom):
    for key,value in custom.items():
        if isinstance(value,dict) and isinstance(base.get(key),dict):_merge(base[key],value)
        else:base[key]=value

def load_config(path:Path)->dict:
    with path.open("r",encoding="utf-8") as stream:custom=yaml.safe_load(stream) or {}
    cfg=deepcopy(DEFAULTS);_merge(cfg,custom)
    for section in ("camera","checkerboard"):
        if section not in cfg:raise ValueError(f"Configuration is missing '{section}'")
    return cfg
