from __future__ import annotations
import cv2
import numpy as np
from scipy.spatial import cKDTree


def camera_corners(image,board,scales):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    for scale in scales:
        test=gray if scale==1 else cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        ok,corners=cv2.findChessboardCornersSB(test,tuple(board),cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY)
        if ok:return corners.reshape(-1,2).astype(np.float64)/scale
    raise RuntimeError(f"Camera checkerboard {board} was not detected")


def _candidate_points(image,cfg):
    smooth=cv2.GaussianBlur(image,(3,3),0)
    corners=cv2.goodFeaturesToTrack(smooth,maxCorners=int(cfg["maximum_corner_candidates"]),qualityLevel=float(cfg["corner_quality"]),
        minDistance=float(cfg["minimum_corner_spacing_px"]),blockSize=5,useHarrisDetector=True,k=.04)
    if corners is None:return np.empty((0,2))
    return corners.reshape(-1,2).astype(np.float64)


def _checker_contrast(image,grid,xvec,yvec,rows,cols):
    """Score alternating diagonal quadrants around all proposed inner corners."""
    q=.30;points=[]
    for sx,sy in ((-1,-1),(1,-1),(-1,1),(1,1)):
        points.append(grid+q*sx*xvec+q*sy*yvec)
    values=[]
    for p in points:
        values.append(cv2.remap(image,p[:,0].astype(np.float32),p[:,1].astype(np.float32),
                    cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).ravel().astype(float))
    response=values[0]+values[3]-values[1]-values[2]
    parity=np.fromiter(((-1.0)**(i+j) for j in range(rows) for i in range(cols)),float,rows*cols)
    # Correct checkerboards produce strong responses with a consistent sign
    # after parity compensation; stripes and arbitrary corner grids do not.
    corner_score=abs(float(np.mean(response*parity)))
    # Also compare complete cell centres. This strongly rejects regular scene
    # structures (windows, racks, panels) that form a lattice but do not
    # alternate black/white like a checkerboard.
    origin=grid[0];centres=[];sign=[]
    for j in range(-1,rows):
        for i in range(-1,cols):
            centres.append(origin+(i+.5)*xvec+(j+.5)*yvec);sign.append((-1.0)**(i+j))
    centres=np.asarray(centres);sample=cv2.remap(cv2.GaussianBlur(image,(5,5),0),
        centres[:,0].astype(np.float32),centres[:,1].astype(np.float32),cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,borderValue=0).ravel().astype(float)
    cell_score=abs(float(np.mean(sample*np.asarray(sign))))
    return corner_score+2.0*cell_score


def lidar_lattice(image,board,cfg):
    """Derivative corner candidates followed by incomplete affine-lattice RANSAC."""
    candidates=_candidate_points(image,cfg);cols,rows=map(int,board)
    if len(candidates)<cfg["minimum_lidar_corners"]:raise RuntimeError("Too few LiDAR intensity corner candidates")
    tree=cKDTree(candidates);rng=np.random.default_rng(int(cfg["random_seed"]));best=None
    anchors=rng.integers(0,len(candidates),size=int(cfg["lattice_trials"]))
    for anchor_id in anchors:
        p=candidates[anchor_id];_,near=tree.query(p,k=min(14,len(candidates)))
        vectors=candidates[np.atleast_1d(near)[1:]]-p
        if len(vectors)<2:continue
        a,b=rng.choice(len(vectors),2,replace=False);vx,vy=vectors[a],vectors[b];sx,sy=np.linalg.norm(vx),np.linalg.norm(vy)
        if min(sx,sy)<3 or max(sx,sy)>28:continue
        cosine=abs(np.dot(vx,vy)/(sx*sy))
        if cosine>.45 or max(sx,sy)/min(sx,sy)>2.0:continue
        xvec,yvec=(vy,vx) if rng.random()<.5 else (vx,vy)
        ai,aj=rng.integers(cols),rng.integers(rows);origin=p-ai*xvec-aj*yvec
        grid=np.array([origin+i*xvec+j*yvec for j in range(rows) for i in range(cols)])
        distance,index=tree.query(grid);tol=float(cfg["lattice_tolerance_fraction"])*min(sx,sy);matched=distance<tol
        contrast=_checker_contrast(image,grid,xvec,yvec,rows,cols)
        score=int(matched.sum())-0.03*float(distance[matched].sum())+0.75*contrast
        if best is None or score>best[0]:best=(score,grid,matched,index,distance)
    if best is None or int(best[2].sum())<int(cfg["minimum_lidar_corners"]):
        raise RuntimeError("No checkerboard-sized lattice found in LiDAR intensity image")
    _,grid,matched,index,_=best
    # Refine the full grid with a homography fitted to matched candidate points.
    ideal=np.array([[i,j] for j in range(rows) for i in range(cols)],np.float64)
    H,_=cv2.findHomography(ideal[matched],candidates[index[matched]],cv2.RANSAC,2.0)
    if H is not None:grid=cv2.perspectiveTransform(ideal.reshape(-1,1,2).astype(np.float32),H).reshape(-1,2)
    distance,index=tree.query(grid);spacing=np.median(np.linalg.norm(np.diff(grid.reshape(rows,cols,2),axis=1),axis=2));matched=distance<float(cfg["lattice_tolerance_fraction"])*spacing
    return grid,matched,candidates


def recover_xyz(grid,matched,projected_uv,point_ids,xyz,radius):
    tree=cKDTree(projected_uv);points=[];indices=[]
    for i,pixel in enumerate(grid):
        if not matched[i]:continue
        ids=tree.query_ball_point(pixel,float(radius))
        if not ids:continue
        points.append(np.median(xyz[point_ids[np.asarray(ids)]],axis=0));indices.append(i)
    return np.asarray(points),np.asarray(indices,dtype=int)
