from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def read_pcd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        fields = sizes = types = counts = None
        points = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            words = text.split()
            if not words or words[0].startswith("#"):
                continue
            key, values = words[0].upper(), words[1:]
            if key == "FIELDS": fields = values
            elif key == "SIZE": sizes = list(map(int, values))
            elif key == "TYPE": types = values
            elif key == "COUNT": counts = list(map(int, values))
            elif key == "POINTS": points = int(values[0])
            elif key == "DATA":
                encoding = values[0].lower()
                break
        if fields is None:
            raise ValueError("PCD FIELDS are missing")
        if encoding == "ascii":
            data = np.loadtxt(f, dtype=np.float64)
            xyz = data[:, [fields.index(k) for k in ("x", "y", "z")]]
            intensity = data[:, fields.index("intensity")] if "intensity" in fields else np.ones(len(data))
        else:
            if encoding != "binary":
                raise ValueError("binary_compressed PCD is not supported; convert it to binary or ASCII PCD")
            counts = counts or [1] * len(fields)
            code = {("F",4):"<f4",("F",8):"<f8",("U",1):"u1",("U",2):"<u2",("U",4):"<u4",("I",1):"i1",("I",2):"<i2",("I",4):"<i4"}
            dtype=[]
            for name,size,kind,count in zip(fields,sizes,types,counts):
                dtype.append((name, code[(kind,size)], (count,)) if count>1 else (name,code[(kind,size)]))
            raw=np.fromfile(f,dtype=np.dtype(dtype),count=points)
            xyz=np.column_stack([raw[k] for k in ("x","y","z")]).astype(np.float64)
            intensity=raw["intensity"].astype(np.float64) if "intensity" in fields else np.ones(len(raw))
    good=np.isfinite(xyz).all(1)&np.isfinite(intensity)
    return xyz[good],intensity[good]


def load_capture(image_path: Path, cloud_path: Path):
    image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f"Cannot read image: {image_path}")
    xyz,intensity=read_pcd(cloud_path)
    return image,xyz,intensity
