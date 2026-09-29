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


def _spherical_pixels(xyz,sph):
    w,h=int(sph["width"]),int(sph["height"])
    f,r,d=(np.asarray(sph[k],float) for k in ("forward_axis","right_axis","down_axis"))
    fw,rt,dn=xyz@f,xyz@r,xyz@d;az=np.arctan2(rt,fw);el=np.arctan2(dn,np.hypot(fw,rt))
    return np.column_stack(((az/np.radians(float(sph["horizontal_fov_deg"]))+.5)*(w-1),(el/np.radians(float(sph["vertical_fov_deg"]))+.5)*(h-1)))


def save_robust(output,image,xyz,board,sph,R,t,K,D,model,image_points,lidar_xyz,metrics):
    """Robust-mode overview: paper panels + board-plane fit + corner check."""
    output.mkdir(parents=True,exist_ok=True);h,w=image.shape[:2]
    uv,front,depth=project_all(xyz,R,t,K,D,model);inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);uv=uv[inside];depth=depth[inside]
    lo,hi=np.percentile(depth,[2,98]);value=np.clip((depth-lo)/max(hi-lo,1e-9)*255,0,255).astype(np.uint8)
    color=cv2.applyColorMap(value.reshape(-1,1),cv2.COLORMAP_TURBO).reshape(-1,3);overlay=image.copy()
    for j in np.argsort(depth)[::-1][::8]:cv2.circle(overlay,tuple(np.rint(uv[j]).astype(int)),1,tuple(map(int,color[j])),-1)
    canvas=image.copy()
    for i,p in enumerate(image_points):cv2.circle(canvas,tuple(np.rint(p).astype(int)),3,(0,255,0),-1)
    # LiDAR spherical image: ROI window (blue), paper lattice (yellow), refined corners (green).
    lat=cv2.cvtColor(board["intensity_image"],cv2.COLOR_GRAY2BGR)
    y0,y1,x0,x1=board["window"];cv2.rectangle(lat,(x0,y0),(x1,y1),(255,160,0),1)
    if board["lattice"] is not None:
        for i,p in enumerate(board["lattice"][0]):cv2.circle(lat,tuple(np.rint(p).astype(int)),2,(0,255,255) if board["lattice"][1][i] else (0,0,255),-1)
    for p in _spherical_pixels(lidar_xyz,sph):cv2.drawMarker(lat,tuple(np.rint(p).astype(int)),(0,255,0),cv2.MARKER_CROSS,6,1)
    lat=lat[max(y0-60,0):y1+60,max(x0-60,0):x1+60]
    # Board-plane pattern fit.
    f=board["fit"];res=f["raster_res"];org=f["raster_origin"]
    fit=cv2.cvtColor(cv2.normalize(f["raster"],None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    s=max(1,int(round(400/max(fit.shape[:2]))));fit=cv2.resize(fit,None,fx=s,fy=s,interpolation=cv2.INTER_NEAREST)
    to_px=lambda p:tuple(np.rint((np.array([(p-f["centroid"])@f["e1"],(p-f["centroid"])@f["e2"]])-org)/res*s).astype(int))
    cv2.polylines(fit,[np.array([to_px(p) for p in f["outline_xyz"]],np.int32)],True,(0,255,0),1)
    for p in lidar_xyz:cv2.circle(fit,to_px(p),3,(0,0,255),-1)
    # Corner check: camera corners (green) vs projected LiDAR corners (red).
    luv=cv2.fisheye.projectPoints((lidar_xyz@R.T+t).reshape(1,-1,3),np.zeros(3),np.zeros(3),K,D.reshape(4,1))[0].reshape(-1,2) if model=="fisheye" \
        else cv2.projectPoints(lidar_xyz@R.T+t,np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
    pts=np.vstack([image_points,luv]);a=np.floor(pts.min(0)).astype(int)-10;b=np.ceil(pts.max(0)).astype(int)+10
    a=np.maximum(a,0);b=np.minimum(b,[w,h]);z=max(1,int(400/max(b-a)))
    chk=cv2.resize(image[a[1]:b[1],a[0]:b[0]],None,fx=z,fy=z,interpolation=cv2.INTER_CUBIC)
    for p in image_points:cv2.circle(chk,tuple(np.rint((p-a)*z+z/2).astype(int)),5,(0,255,0),1)
    for p in luv:cv2.drawMarker(chk,tuple(np.rint((p-a)*z+z/2).astype(int)),(0,0,255),cv2.MARKER_CROSS,10,1)
    status=metrics["quality_status"].upper();colour=(40,200,40) if status=="PASS" else (0,165,255)
    header=np.full((110,1200,3),24,np.uint8)
    cv2.putText(header,f"Calibration {status}  (robust mode)",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.9,colour,2,cv2.LINE_AA)
    cv2.putText(header,f"E-PnP {metrics['initial_rms']:.3f} px -> LM {metrics['rms']:.3f} px | {metrics['corners']} corners | "
                f"board {metrics['board_range_m']:.2f} m, square {metrics['square_mm']:.1f} mm",(20,68),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),1,cv2.LINE_AA)
    cv2.putText(header,"; ".join(metrics["notes"])[:150] or "all quality gates passed",(20,96),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,200,200),1,cv2.LINE_AA)
    rows=[np.hstack((_label(_fit_panel(canvas),"Camera checkerboard corners"),_label(_fit_panel(lat),"LiDAR: ROI, paper lattice, refined corners"))),
          np.hstack((_label(_fit_panel(fit),"Board-plane pattern fit (LiDAR)"),_label(_fit_panel(chk),"Corner check: camera (green) vs LiDAR (red)"))),
          np.hstack((_label(_fit_panel(image),"Camera image (temporal median)"),_label(_fit_panel(overlay),"LiDAR projected into camera")))]
    ok,buf=cv2.imencode(".png",np.vstack([header]+rows))
    (output/"calibration_overview.png").write_bytes(buf.tobytes())
