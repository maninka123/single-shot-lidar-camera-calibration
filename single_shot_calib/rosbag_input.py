"""Optional ROS bag extraction.

ROS packages are intentionally not installed by pip. Run this module inside the
ROS1/ROS2 Python environment matching the bag. Messages are synchronized by their
record timestamps; point clouds inside merge_window_sec are concatenated because
the calibration rig must remain stationary during one capture.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def _choose(images,clouds,merge_window,sync_tolerance):
    if not images or not clouds:raise RuntimeError("Configured ROS topics contained no usable messages")
    cloud_center=(clouds[0][0]+clouds[-1][0])/2
    center,image=min(images,key=lambda item:abs(item[0]-cloud_center))
    nearest=min(abs(stamp-center) for stamp,_ in clouds)
    if nearest>sync_tolerance:raise RuntimeError(f"Nearest image/cloud timestamps differ by {nearest:.3f}s (limit {sync_tolerance:.3f}s)")
    selected=[points for stamp,points in clouds if abs(stamp-center)<=merge_window/2]
    if not selected:raise RuntimeError("No point clouds fall inside merge_window_sec")
    data=np.concatenate(selected);return image,data[:,:3],data[:,3] if data.shape[1]>3 else np.ones(len(data))


def _image(msg,bridge):
    if msg.__class__.__name__=="CompressedImage":
        decoded=cv2.imdecode(np.frombuffer(msg.data,dtype=np.uint8),cv2.IMREAD_COLOR)
        if decoded is None:raise RuntimeError("Could not decode CompressedImage")
        return decoded
    return bridge.imgmsg_to_cv2(msg,'bgr8')


def extract_ros1(bag_path:Path,image_topic:str,cloud_topic:str,merge_window:float,sync_tolerance:float):
    try:
        import rosbag
        from cv_bridge import CvBridge
        from sensor_msgs import point_cloud2
    except ImportError as e:raise RuntimeError("ROS1 input requires rosbag, cv_bridge, and sensor_msgs in the active ROS environment") from e
    bridge=CvBridge();images=[];clouds=[]
    with rosbag.Bag(str(bag_path)) as bag:
        for topic,msg,t in bag.read_messages(topics=[image_topic,cloud_topic]):
            stamp=t.to_sec()
            if topic==image_topic:images.append((stamp,_image(msg,bridge)))
            else:
                fields=[f.name for f in msg.fields];pts=np.asarray(list(point_cloud2.read_points(msg,field_names=tuple(x for x in ('x','y','z','intensity') if x in fields),skip_nans=True)))
                clouds.append((stamp,pts))
    return _choose(images,clouds,merge_window,sync_tolerance)


def extract_ros2(bag_dir:Path,image_topic:str,cloud_topic:str,merge_window:float,sync_tolerance:float):
    try:
        import rosbag2_py
        from cv_bridge import CvBridge
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from sensor_msgs_py import point_cloud2
    except ImportError as e:raise RuntimeError("ROS2 input requires a sourced ROS2 environment with rosbag2_py, rclpy, cv_bridge, and sensor_msgs_py") from e
    storage_id='mcap' if any(bag_dir.glob('*.mcap')) else 'sqlite3';reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir),storage_id=storage_id),rosbag2_py.ConverterOptions(input_serialization_format='cdr',output_serialization_format='cdr'))
    types={entry.name:entry.type for entry in reader.get_all_topics_and_types()}
    missing=[x for x in (image_topic,cloud_topic) if x not in types]
    if missing:raise RuntimeError(f"ROS2 bag is missing topics: {missing}")
    classes={x:get_message(types[x]) for x in (image_topic,cloud_topic)};bridge=CvBridge();images=[];clouds=[]
    while reader.has_next():
        topic,serialized,stamp_ns=reader.read_next()
        if topic not in classes:continue
        msg=deserialize_message(serialized,classes[topic]);stamp=stamp_ns*1e-9
        if topic==image_topic:images.append((stamp,_image(msg,bridge)))
        else:
            names={f.name for f in msg.fields};wanted=[x for x in ('x','y','z','intensity') if x in names]
            points=np.asarray(list(point_cloud2.read_points(msg,field_names=wanted,skip_nans=True)),dtype=float)
            if len(points):clouds.append((stamp,points))
    return _choose(images,clouds,merge_window,sync_tolerance)


# ---------------------------------------------------------------- robust mode
def _decode_image_raw(msg):
    data=np.frombuffer(msg.data,dtype=np.uint8)
    if msg.__class__.__name__.endswith("CompressedImage") or not hasattr(msg,"encoding"):
        return cv2.imdecode(data,cv2.IMREAD_COLOR)
    enc=msg.encoding.lower()
    if enc in ("rgb8","bgr8"):
        img=data.reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
        return img[:,:,::-1].copy() if enc=="rgb8" else img.copy()
    if enc in ("mono8","8uc1"):
        return cv2.cvtColor(data.reshape(msg.height,msg.step)[:,:msg.width].copy(),cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Unsupported image encoding {msg.encoding}")


def _decode_cloud_raw(msg):
    off={f.name:f.offset for f in msg.fields};rec=np.frombuffer(msg.data,dtype=np.uint8).reshape(-1,msg.point_step)
    out=np.ones((len(rec),4),np.float32)
    for i,name in enumerate(("x","y","z","intensity")):
        if name in off:out[:,i]=rec[:,off[name]:off[name]+4].copy().view("<f4").ravel()
    return out


def read_bag_frames(bag_path:Path,image_topic:str,cloud_topic:str):
    """All image and cloud messages of one capture (record time).  Uses the pure-Python
    `rosbags` package (ROS1 .bag or ROS2 directory, no ROS installation needed)."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as e:raise RuntimeError("Robust bag input needs `pip install rosbags`") from e
    images=[];clouds=[]
    with AnyReader([Path(bag_path)]) as reader:
        conns=[c for c in reader.connections if c.topic in (image_topic,cloud_topic)]
        missing={image_topic,cloud_topic}-{c.topic for c in conns}
        if missing:raise RuntimeError(f"Bag is missing topics: {sorted(missing)}")
        for con,stamp,raw in reader.messages(connections=conns):
            msg=reader.deserialize(raw,con.msgtype)
            if con.topic==image_topic:images.append((stamp*1e-9,_decode_image_raw(msg)))
            else:clouds.append((stamp*1e-9,_decode_cloud_raw(msg)))
    if not images or not clouds:raise RuntimeError("Configured topics contained no messages")
    return images,clouds


def extract_robust(bag_path:Path,image_topic:str,cloud_topic:str,robust_cfg:dict,merge_window:float):
    """Stationary capture -> (median image, merged xyz, intensity, per-point frame index, info).

    The whole recording is used when robust.merge_full_capture is true (the rig must be
    stationary, which is checked on the images here and on the board later); otherwise
    the paper's merge_window_sec around the centre of the recording."""
    from .robust import median_image
    images,clouds=read_bag_frames(bag_path,image_topic,cloud_topic)
    if not robust_cfg.get("merge_full_capture",True):
        centre=(clouds[0][0]+clouds[-1][0])/2
        clouds=[c for c in clouds if abs(c[0]-centre)<=merge_window/2]
        images=[i for i in images if abs(i[0]-centre)<=merge_window/2] or [min(images,key=lambda i:abs(i[0]-centre))]
    t0,t1=clouds[0][0],clouds[-1][0]
    frames=[img for stamp,img in images if t0-0.1<=stamp<=t1+0.1] or [img for _,img in images]
    if robust_cfg.get("camera_median",True):image,info=median_image(frames,float(robust_cfg["maximum_frame_deviation"]))
    else:image,info=frames[len(frames)//2],dict(frames=len(frames),frames_used=1)
    data=np.concatenate([c for _,c in clouds]);frame=np.concatenate([np.full(len(c),i) for i,(_,c) in enumerate(clouds)])
    good=np.isfinite(data).all(1)&(np.linalg.norm(data[:,:3],axis=1)>0.05)
    info.update(lidar_frames=len(clouds),lidar_seconds=float(t1-t0))
    return image,data[good,:3].astype(np.float64),data[good,3].astype(np.float64),frame[good],info
