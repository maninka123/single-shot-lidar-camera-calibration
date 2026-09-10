from __future__ import annotations
import cv2
import numpy as np


def project_all(xyz,R,t,K,D,model):
    camera=xyz@R.T+t;front=camera[:,2]>0
    if model=='fisheye':uv=cv2.fisheye.projectPoints(camera[front].reshape(1,-1,3),np.zeros(3),np.zeros(3),K,D.reshape(4,1))[0].reshape(-1,2)
    else:uv=cv2.projectPoints(camera[front],np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
    return uv,front,camera[front,2]


def _fit_panel(image,size=(600,450)):
    width,height=size;scale=min(width/image.shape[1],height/image.shape[0]);resized=cv2.resize(image,None,fx=scale,fy=scale)
    panel=np.full((height,width,3),24,np.uint8);y=(height-resized.shape[0])//2;x=(width-resized.shape[1])//2
    panel[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return panel


def _label(image,text):
    cv2.rectangle(image,(0,0),(image.shape[1],34),(15,15,15),-1)
    cv2.putText(image,text,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
    return image


def save(output,image,xyz,lidar_intensity_image,R,t,K,D,model,camera_points,lidar_grid,matched,candidates,metrics):
    output.mkdir(parents=True,exist_ok=True);h,w=image.shape[:2];uv,front,depth=project_all(xyz,R,t,K,D,model);inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);uv=uv[inside];depth=depth[inside]
    lo,hi=np.percentile(depth,[2,98]);value=np.clip((depth-lo)/max(hi-lo,1e-9)*255,0,255).astype(np.uint8);color=cv2.applyColorMap(value.reshape(-1,1),cv2.COLORMAP_TURBO).reshape(-1,3);overlay=image.copy()
    # Sparse points keep the camera scene visible enough for alignment checks.
    for j in np.argsort(depth)[::-1][::8]:cv2.circle(overlay,tuple(np.rint(uv[j]).astype(int)),1,tuple(map(int,color[j])),-1)
    canvas=image.copy()
    for i,p in enumerate(camera_points):cv2.circle(canvas,tuple(np.rint(p).astype(int)),3,(0,255,0),-1);cv2.putText(canvas,str(i),tuple(np.rint(p+[3,-3]).astype(int)),cv2.FONT_HERSHEY_PLAIN,.6,(0,255,0),1)
    lattice=cv2.cvtColor(lidar_intensity_image,cv2.COLOR_GRAY2BGR)
    for p in candidates:cv2.circle(lattice,tuple(np.rint(p).astype(int)),1,(80,80,80),-1)
    for i,p in enumerate(lidar_grid):cv2.circle(lattice,tuple(np.rint(p).astype(int)),2,(0,255,0) if matched[i] else (0,0,255),-1)
    status=metrics['quality_status'].upper();colour=(40,200,40) if status=='PASS' else (0,165,255)
    header=np.full((90,1200,3),24,np.uint8)
    cv2.putText(header,f"Calibration {status}",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.9,colour,2,cv2.LINE_AA)
    cv2.putText(header,f"E-PnP {metrics['initial_rms']:.3f} px  ->  LM {metrics['rms']:.3f} px   |   LiDAR corners {metrics['corners']}",(20,72),cv2.FONT_HERSHEY_SIMPLEX,.65,(230,230,230),2,cv2.LINE_AA)
    top=np.hstack((_label(_fit_panel(canvas),"Camera checkerboard corners"),_label(_fit_panel(lattice),"LiDAR intensity lattice")))
    bottom=np.hstack((_label(_fit_panel(image),"Camera image"),_label(_fit_panel(overlay),"LiDAR projected into camera")))
    cv2.imwrite(str(output/'calibration_overview.png'),np.vstack((header,top,bottom)))
