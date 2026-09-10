from copy import deepcopy
from pathlib import Path
import yaml

DEFAULTS={
 "input":{"mode":"auto","image":"data/image.png","pointcloud":"data/cloud.pcd","pairs_dir":None,"image_dir":None,"pointcloud_dir":None,"bags":[],"image_topic":"/camera/image_raw","pointcloud_topic":"/livox/lidar","sync_tolerance_sec":0.05,"merge_window_sec":1.0},
 "checkerboard":{"square_size_mm":None},
 "spherical_projection":{"width":600,"height":600,"horizontal_fov_deg":90.0,"vertical_fov_deg":90.0,"forward_axis":[1,0,0],"right_axis":[0,-1,0],"down_axis":[0,0,-1],"minimum_range_m":0.35,"maximum_range_m":20.0},
 "detection":{"camera_upscale_factors":[1,2,3,4,5,6],"maximum_corner_candidates":1800,"corner_quality":0.003,"minimum_corner_spacing_px":3.0,"lattice_trials":5000,"lattice_tolerance_fraction":0.38,"minimum_lidar_corners":18,"point_recovery_radius_px":2.5,"random_seed":17},
 "optimization":{"maximum_iterations":300,"ftol":1e-10,"xtol":1e-10,"gtol":1e-10},
 "quality":{"maximum_reprojection_rmse_px":3.0},"output_dir":"outputs"}

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
