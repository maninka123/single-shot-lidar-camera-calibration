from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares


def project_camera(points,rvec,tvec,K,D,model):
    obj=np.asarray(points,np.float64).reshape(1,-1,3)
    if model=="fisheye":return cv2.fisheye.projectPoints(obj,rvec,tvec,K,D.reshape(4,1))[0].reshape(-1,2)
    return cv2.projectPoints(obj,rvec,tvec,K,D)[0].reshape(-1,2)


def normalized_camera_points(points,K,D,model):
    p=np.asarray(points,np.float64).reshape(-1,1,2)
    return (cv2.fisheye.undistortPoints(p,K,D.reshape(4,1)) if model=="fisheye" else cv2.undistortPoints(p,K,D)).reshape(-1,2)


def solve(lidar_xyz,lattice_indices,camera_grid,board,K,D,model,opt_cfg):
    cols,rows=map(int,board);base=camera_grid.reshape(rows,cols,2);solutions=[]
    variants=[("identity",base),("flip_x",base[:,::-1]),("flip_y",base[::-1]),("flip_xy",base[::-1,::-1])]
    for name,array in variants:
        image_points=array.reshape(-1,2)[lattice_indices]
        normalized=normalized_camera_points(image_points,K,D,model)
        ok,rvec,tvec=cv2.solvePnP(lidar_xyz,normalized,np.eye(3),None,flags=cv2.SOLVEPNP_EPNP)
        if not ok:continue
        def residual(x):return (project_camera(lidar_xyz,x[:3],x[3:],K,D,model)-image_points).ravel()
        x0=np.r_[rvec.ravel(),tvec.ravel()]
        initial_errors=np.linalg.norm(residual(x0).reshape(-1,2),axis=1)
        # The paper uses Levenberg-Marquardt after E-PnP.  Residuals are kept
        # in distorted camera pixels so the reported error is directly useful.
        fit=least_squares(residual,x0,method="lm",max_nfev=int(opt_cfg["maximum_iterations"]),
                          ftol=float(opt_cfg.get("ftol",1e-10)),xtol=float(opt_cfg.get("xtol",1e-10)),
                          gtol=float(opt_cfg.get("gtol",1e-10)))
        rotation=cv2.Rodrigues(fit.x[:3])[0];camera_xyz=lidar_xyz@rotation.T+fit.x[3:]
        if np.mean(camera_xyz[:,2]>0)<.9:continue
        errors=np.linalg.norm(residual(fit.x).reshape(-1,2),axis=1)
        solutions.append((float(np.sqrt(np.mean(errors**2))),name,rotation,fit.x[3:],image_points,errors,
                          float(np.sqrt(np.mean(initial_errors**2)))))
    if not solutions:raise RuntimeError("E-PnP did not produce a physically valid camera pose")
    return min(solutions,key=lambda x:x[0])


def solve_robust(lidar_xyz,lattice_indices,camera_grid,board,K,D,model,opt_cfg,robust_cfg,board_normal=None):
    """Robust-mode solver: the paper's E-PnP + LM for every lattice ordering, then

    * mirrored orderings are rejected (the board must face both sensors),
    * a symmetric (odd x odd squares) board keeps two equally good solutions 180 deg
      apart; `approx_rotation_lidar_to_camera` selects between them,
    * LM runs with a robust loss and reports a covariance-based uncertainty.
    Returns (best solution dict, list of all solutions, notes)."""
    cols,rows=map(int,board);base=camera_grid.reshape(rows,cols,2);notes=[];sols=[]
    variants=[("identity",base),("flip_x",base[:,::-1]),("flip_y",base[::-1]),("flip_xy",base[::-1,::-1])]
    loss=robust_cfg.get("loss","linear");scale=float(robust_cfg.get("loss_scale_px",1.0))
    for name,array in variants:
        image_points=array.reshape(-1,2)[lattice_indices]
        normalized=normalized_camera_points(image_points,K,D,model)
        ok,rvec,tvec=cv2.solvePnP(lidar_xyz,normalized,np.eye(3),None,flags=cv2.SOLVEPNP_EPNP)
        if not ok:continue
        def residual(x):return (project_camera(lidar_xyz,x[:3],x[3:],K,D,model)-image_points).ravel()
        x0=np.r_[rvec.ravel(),tvec.ravel()];initial=np.linalg.norm(residual(x0).reshape(-1,2),axis=1)
        fit=least_squares(residual,x0,method="lm",max_nfev=int(opt_cfg["maximum_iterations"]))  # paper LM
        if loss!="linear":fit=least_squares(residual,fit.x,method="trf",loss=loss,f_scale=scale,max_nfev=int(opt_cfg["maximum_iterations"]))
        R=cv2.Rodrigues(fit.x[:3])[0];t=fit.x[3:];cam=lidar_xyz@R.T+t
        errors=np.linalg.norm(residual(fit.x).reshape(-1,2),axis=1);rms=float(np.sqrt(np.mean(errors**2)))
        facing=True
        if board_normal is not None:
            n=board_normal/np.linalg.norm(board_normal);centre=lidar_xyz.mean(0)
            if n@centre>0:n=-n  # towards the LiDAR
            facing=bool((R@n)@(R@centre+t)<0)
        # Corner noise is never below the configured floor: a near-perfect fit of two ideal
        # grids would otherwise report an unrealistically small uncertainty.
        dof=max(2*len(errors)-6,1);J=fit.jac;floor=float(robust_cfg.get("uncertainty_noise_floor_px",0.25))
        try:cov=np.linalg.inv(J.T@J)*max(float(np.sum(residual(fit.x)**2)/dof),floor**2)
        except np.linalg.LinAlgError:cov=np.full((6,6),np.nan)
        sd=np.sqrt(np.clip(np.diag(cov),0,None))
        sols.append(dict(variant=name,rms=rms,initial_rms=float(np.sqrt(np.mean(initial**2))),R=R,t=t,errors=errors,
            image_points=image_points,in_front=bool(np.mean(cam[:,2]>0)>=.9),facing=facing,
            rotation_std_deg=float(np.degrees(np.linalg.norm(sd[:3]))),translation_std_mm=float(np.linalg.norm(sd[3:])*1000)))
    valid=[s for s in sols if s["in_front"] and (s["facing"] or not robust_cfg.get("reject_mirrored",True))]
    if not valid:raise RuntimeError("E-PnP did not produce a physically valid camera pose")
    prior=robust_cfg.get("approx_rotation_lidar_to_camera")
    if prior is not None:
        P=np.asarray(prior,float);tol=float(robust_cfg.get("approx_rotation_tolerance_deg",45))
        for s in valid:s["prior_angle_deg"]=_angle_deg(s["R"],P)
        near=[s for s in valid if s["prior_angle_deg"]<=tol]
        if near:valid=near
        else:notes.append(f"no solution within {tol:.0f} deg of approx_rotation_lidar_to_camera")
    best=min(valid,key=lambda s:s["rms"])
    rivals=[s for s in valid if s is not best and s["rms"]<=1.5*best["rms"]+.05 and _angle_deg(s["R"],best["R"])>20]
    if rivals and ((cols+1)%2==1 and (rows+1)%2==1):
        notes.append("symmetric board (odd x odd squares): a 180-degree-rotated solution fits equally well; "
                     "set robust.approx_rotation_lidar_to_camera or use an asymmetric board")
    return best,sols,notes


def _angle_deg(Ra,Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra@Rb.T)-1)/2,-1,1))))
