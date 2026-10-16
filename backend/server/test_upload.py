"""
Send a synthetic scan to the running server - checks the upload path without the phone.
A 4 x 3 m floor (y = -1.4, camera held ~1.4 m above it), two walls, a table top, 3 keyframes.

Usage: python backend/server/test_upload.py [http://127.0.0.1:8000]
"""
import io
import json
import math
import sys

import requests


def rect(x0, z0, x1, z1, y):
    return [x0, y, z0, x1, y, z0, x1, y, z1, x0, y, z1]


def pose(x, y, z, yaw_deg):
    """Unity localToWorld (column-major), camera looking along its +Z rotated by yaw."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    # columns: right, up, forward, position
    return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, x, y, z, 1]


def tiny_jpeg():
    # 1x1 white JPEG
    return bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c"
        "20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffc0000b080001000101011100ffc4001f00000105010101010101000000"
        "0000000000010203040506070809000a0bffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191"
        "a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a73"
        "7475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9da"
        "e1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda0008010100003f00fbfcffd9")


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    floor_y = -1.4
    scan = {
        "app_version": "test", "device": "synthetic", "started_at": "2026-10-15 00:00:00",
        "duration_s": 30.0, "screen_orientation": "Portrait", "tracking_lost_count": 0,
        "planes": [
            {"id": "floor1", "alignment": "HorizontalUp", "center": [0, floor_y, 0], "normal": [0, 1, 0],
             "size": [4, 3], "boundary": rect(-2, -1.5, 2, 1.5, floor_y)},
            {"id": "table", "alignment": "HorizontalUp", "center": [1.2, floor_y + 0.75, 0.8], "normal": [0, 1, 0],
             "size": [1.0, 0.6], "boundary": rect(0.7, 0.5, 1.7, 1.1, floor_y + 0.75)},
            {"id": "wallN", "alignment": "Vertical", "center": [0, 0, 1.5], "normal": [0, 0, -1],
             "size": [4, 2.5], "boundary": [-2, floor_y, 1.5, 2, floor_y, 1.5, 2, 1.1, 1.5, -2, 1.1, 1.5]},
            {"id": "wallW", "alignment": "Vertical", "center": [-2, 0, 0], "normal": [1, 0, 0],
             "size": [3, 2.5], "boundary": [-2, floor_y, -1.5, -2, floor_y, 1.5, -2, 1.1, 1.5, -2, 1.1, -1.5]},
        ],
        "keyframes": [
            {"file": f"kf_{i:03}.jpg", "t": i * 2.0, "pose": pose(x, 0.0, z, yaw),
             "fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480}
            for i, (x, z, yaw) in enumerate([(0, -1, 0), (0.6, -0.8, 30), (1.0, -0.4, 60)])
        ],
    }
    files = [("files", (k["file"], io.BytesIO(tiny_jpeg()), "image/jpeg")) for k in scan["keyframes"]]
    r = requests.post(url + "/scans", data={"scan": json.dumps(scan)}, files=files, timeout=30)
    print(r.status_code, json.dumps(r.json(), ensure_ascii=False, indent=1))
    expect = {"floor_area_m2": 12.0, "floor_planes": 1, "wall_planes": 2, "keyframes": 3, "camera_height_above_floor_m": 1.4}
    bad = {k: (r.json().get(k), v) for k, v in expect.items() if r.json().get(k) != v}
    print("CHECK:", "PASS" if not bad else f"FAIL {bad}")


if __name__ == "__main__":
    main()
