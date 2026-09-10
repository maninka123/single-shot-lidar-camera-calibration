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
