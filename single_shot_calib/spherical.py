from __future__ import annotations
import cv2
import numpy as np


def project(xyz: np.ndarray, intensity: np.ndarray, cfg: dict):
    width,height=int(cfg["width"]),int(cfg["height"])
    forward=xyz@np.asarray(cfg["forward_axis"],float);right=xyz@np.asarray(cfg["right_axis"],float);down=xyz@np.asarray(cfg["down_axis"],float)
    ranges=np.linalg.norm(xyz,axis=1);az=np.arctan2(right,forward);el=np.arctan2(down,np.hypot(forward,right))
    hfov=np.radians(float(cfg["horizontal_fov_deg"]));vfov=np.radians(float(cfg["vertical_fov_deg"]))
    u=np.rint((az/hfov+.5)*(width-1)).astype(int);v=np.rint((el/vfov+.5)*(height-1)).astype(int)
    keep=(forward>0)&(ranges>=cfg["minimum_range_m"])&(ranges<=cfg["maximum_range_m"])&(u>=0)&(u<width)&(v>=0)&(v<height)
    ids=np.flatnonzero(keep);u,v=u[keep],v[keep]
    sums=np.zeros((height,width),np.float64);counts=np.zeros((height,width),np.int32)
    np.add.at(sums,(v,u),intensity[keep]);np.add.at(counts,(v,u),1)
    raw=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0);valid=counts>0
    lo,hi=np.percentile(raw[valid],[2,99]) if valid.any() else (0,1)
    image=np.clip((raw-lo)/max(hi-lo,1e-9)*255,0,255).astype(np.uint8)
    image=cv2.createCLAHE(3.0,(8,8)).apply(image)
    return image,np.column_stack((u,v)).astype(np.float64),ids,valid

