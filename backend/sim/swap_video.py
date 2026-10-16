"""
Step 8b - turn an ARCore recording of the REAL camera into one that shows the VIRTUAL room.

ARCore datasets (rec.mp4) hold the camera video plus IMU/metadata tracks. ARCore playback re-runs
its own tracking and plane detection on whatever video the file contains, so replacing the video
with the virtual room rendered from the same poses (frames/v_XXXXX.jpg, see SimRecorder.cs) makes
ARCore 'scan' the virtual room with its real algorithm.

Usage:
  python backend/sim/swap_video.py pull                 # adb pull every new recording -> data/sim/recordings/
  python backend/sim/swap_video.py probe <rec_id>       # list the tracks of rec.mp4
  python backend/sim/swap_video.py make <rec_id>        # -> data/sim/playback/<rec>_{1copy,2reencode,3virtual}.mp4
  python backend/sim/swap_video.py push <rec_id>        # adb push them to sim/playback on the phone
"""

import json
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mp4box as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "data" / "sim" / "recordings"
ADB = Path.home() / "AppData" / "Local" / "Android" / "Sdk" / "platform-tools" / "adb.exe"
PHONE_SIM = "/sdcard/Android/data/com.doha.ar/files/sim"


def adb(*args, check=True):
    return subprocess.run([str(ADB), *args], capture_output=True, text=True, check=check).stdout


def pull():
    REC.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in adb("shell", "ls", PHONE_SIM).split() if n.strip().startswith("rec_")]
    for n in names:
        if (REC / n / "rec.mp4").exists():
            continue
        print("pull", n)
        adb("pull", f"{PHONE_SIM}/{n}", str(REC))
    reports = ROOT / "data" / "sim" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    adb("pull", f"{PHONE_SIM}/reports/.", str(reports), check=False)
    return names


def probe(rec_id):
    import av
    path = REC / rec_id / "rec.mp4"
    out = {"file": str(path), "bytes": path.stat().st_size, "streams": []}
    with av.open(str(path)) as c:
        out["format"] = c.format.name
        out["metadata"] = dict(c.metadata)
        for s in c.streams:
            info = {"index": s.index, "type": s.type, "codec": getattr(s.codec_context, "name", None) if s.codec_context else None,
                    "time_base": str(s.time_base), "frames": s.frames, "duration": s.duration,
                    "metadata": dict(s.metadata)}
            if s.type == "video":
                info.update(width=s.codec_context.width, height=s.codec_context.height,
                            pix_fmt=s.codec_context.pix_fmt, profile=s.codec_context.profile,
                            avg_rate=str(s.average_rate))
            out["streams"].append(info)
    return out


# ---------------------------------------------------------------- frame timestamps
def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            return r, i


def video_timestamps(data, moov):
    """Sensor timestamp (ns) of every video frame, from the protobuf samples of the
    'application/arcore-video-0' track (field 2.3) - same clock as ARCameraFrameEventArgs.timestampNs."""
    tr = {e: t for _, e, t in M.tracks(moov)}
    out = []
    for off, n in M.sample_offsets(M.stbl_of(tr["application/arcore-video-0"])):
        b = data[off:off + n]
        outer, i = [], 0
        _, i = _varint(b, i)                      # field 1 (empty)
        n1, i = _varint(b, i)
        i += n1
        _, i = _varint(b, i)                      # field 2 (message)
        n2, i = _varint(b, i)
        msg, j, ts = b[i:i + n2], 0, None
        while j < len(msg):
            key, j = _varint(msg, j)
            f, w = key >> 3, key & 7
            if w == 0:
                v, j = _varint(msg, j)
                if f == 3:
                    ts = v
                    break
            elif w == 2:
                ln, j = _varint(msg, j)
                j += ln
            elif w == 1:
                j += 8
            elif w == 5:
                j += 4
        out.append(ts)
    return out


# ---------------------------------------------------------------- H.264 helpers
def nal_units(annexb):
    """Split an Annex-B byte stream into NAL units (without start codes)."""
    import re
    starts = [m.end() for m in re.finditer(b"\x00\x00\x01", annexb)]
    nals = []
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else len(annexb)
        nal = annexb[s:e]
        while nal.endswith(b"\x00") and k + 1 < len(starts):
            nal = nal[:-1]                         # 4-byte start code of the next NAL
        nals.append(nal)
    return nals


def avcc_box(sps, pps):
    b = bytes([1, sps[1], sps[2], sps[3], 0xFF, 0xE1]) + struct.pack(">H", len(sps)) + sps
    b += bytes([1]) + struct.pack(">H", len(pps)) + pps
    if sps[1] in (100, 110, 122, 144):            # High profiles: chroma 4:2:0, 8 bit, no SPS ext
        b += bytes([0xFC | 1, 0xF8, 0xF8, 0])
    return struct.pack(">I4s", 8 + len(b), b"avcC") + b


def encode(frames_rgb, width, height):
    """libx264, no B-frames (as ARCore's encoder), keyframe every 33 frames. -> (samples, keyframes, avcC)."""
    import av
    from fractions import Fraction
    cc = av.CodecContext.create("libx264", "w")
    cc.width, cc.height, cc.pix_fmt = width, height, "yuv420p"
    cc.time_base = Fraction(1, 30)
    cc.framerate = Fraction(30, 1)
    cc.options = {"profile": "high", "preset": "medium", "tune": "zerolatency", "crf": "18",
                  "bframes": "0", "g": "33", "keyint_min": "33", "sc_threshold": "0"}
    samples, keys, sps, pps = [], [], None, None

    def take(packets):
        nonlocal sps, pps
        for p in packets:
            body = b""
            for nal in nal_units(bytes(p)):
                t = nal[0] & 0x1F
                if t == 7:
                    sps = sps or nal
                elif t == 8:
                    pps = pps or nal
                elif t in (6, 9):                  # SEI, access unit delimiter: not needed
                    continue
                else:
                    body += struct.pack(">I", len(nal)) + nal
            samples.append(body)
            keys.append(bool(p.is_keyframe))

    for i, rgb in enumerate(frames_rgb):
        fr = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
        fr.pts = i
        take(cc.encode(fr))
    take(cc.encode(None))
    return samples, keys, avcc_box(sps, pps)


# ---------------------------------------------------------------- dataset writer
def write_dataset(src_data, moov_src, out_path, samples, keys=None, avcc=None):
    """Same file, but the video track points at `samples` (one chunk per sample, appended in a new mdat).
    Everything else (IMU/metadata samples in the old mdat, their tables, timing, uuid boxes) is untouched."""
    moov = M.parse(moov_src)[0]
    video = next(t for h, _, t in M.tracks(moov) if h == "vide")
    stbl = M.stbl_of(video)
    n_old = len(M.read_stsz(stbl.find(b"stsz").payload))
    assert len(samples) == n_old, f"{len(samples)} samples, track has {n_old}"

    prefix = src_data[:moov_offset(src_data)]
    base = len(prefix) + 8
    offsets, pos = [], base
    for s in samples:
        offsets.append(pos)
        pos += len(s)
    mdat = struct.pack(">I4s", 8 + sum(map(len, samples)), b"mdat") + b"".join(samples)

    def full(body):
        return b"\0\0\0\0" + body
    for c in stbl.children:
        if c.type == b"stsz":
            c.payload = full(struct.pack(f">II{len(samples)}I", 0, len(samples), *map(len, samples)))
        elif c.type == b"stsc":
            c.payload = full(struct.pack(">IIII", 1, 1, 1, 1))
        elif c.type in (b"stco", b"co64"):
            c.type, c.payload = b"stco", full(struct.pack(f">I{len(offsets)}I", len(offsets), *offsets))
        elif c.type == b"stss" and keys is not None:
            idx = [i + 1 for i, k in enumerate(keys) if k]
            c.payload = full(struct.pack(f">I{len(idx)}I", len(idx), *idx))
        elif c.type == b"stsd" and avcc is not None:
            p = c.payload
            entry = p[8:]
            size, typ = struct.unpack_from(">I4s", entry)
            a = entry.index(b"avcC") - 4
            old_len = struct.unpack_from(">I", entry, a)[0]
            new_entry = entry[:a] + avcc + entry[a + old_len:size]
            new_entry = struct.pack(">I", len(new_entry)) + new_entry[4:]
            c.payload = p[:8] + new_entry + entry[size:]
    Path(out_path).write_bytes(prefix + mdat + moov.serialize())


def moov_offset(data):
    off = 0
    while off < len(data):
        size, typ = struct.unpack_from(">I4s", data, off)
        if size == 1:
            size = struct.unpack_from(">Q", data, off + 8)[0]
        if typ == b"moov":
            assert off + size == len(data), "moov must be the last box"
            return off
        off += size
    raise ValueError("no moov")


def global_flow(gray_frames):
    """Frame-to-frame image shift (phase correlation) -> array (n-1, 3): dx, dy, response."""
    import cv2
    import numpy as np
    out = []
    for a, b in zip(gray_frames, gray_frames[1:]):
        (dx, dy), r = cv2.phaseCorrelate(a, b)
        out.append((dx, dy, r))
    return np.array(out)


def estimate_lag(real_flow, virt_flow, lags=range(-3, 7)):
    """Virtual frames are rendered with the pose Unity has in LateUpdate, which on the A55 is
    2 camera frames older than the frame timestamp (rec_20261016_095026: motion correlation
    0.80-0.96 at lag 2 vs 0.36 at lag 0). Find the lag that best matches the image motion."""
    import numpy as np
    best = (None, -2)
    for lag in lags:
        n = min(len(real_flow), len(virt_flow)) - abs(lag) - 10
        x = real_flow[5:5 + n]
        y = virt_flow[5 + lag:5 + lag + n]
        m = (x[:, 2] > 0.1) & (y[:, 2] > 0.1)
        c = (np.corrcoef(x[m, 0], y[m, 0])[0, 1] + np.corrcoef(x[m, 1], y[m, 1])[0, 1]) / 2
        if c > best[1]:
            best = (lag, c)
    return best


def make(rec_id, variants=("copy", "reencode", "virtual")):
    import av
    import numpy as np
    from PIL import Image
    folder = REC / rec_id
    data = (folder / "rec.mp4").read_bytes()
    mo = moov_offset(data)
    moov_src = data[mo:]
    moov = M.parse(moov_src)[0]
    video = next(t for h, _, t in M.tracks(moov) if h == "vide")
    orig = [data[o:o + n] for o, n in M.sample_offsets(M.stbl_of(video))]
    vts = np.array(video_timestamps(data, moov), dtype=np.int64)
    out_dir = ROOT / "data" / "sim" / "playback"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"rec": rec_id, "video_frames": len(orig)}

    if "copy" in variants:
        write_dataset(data, moov_src, out_dir / f"{rec_id}_1copy.mp4", orig)

    if "reencode" in variants:
        with av.open(str(folder / "rec.mp4")) as c:
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        h, w = frames[0].shape[:2]
        s, k, a = encode(frames, w, h)
        write_dataset(data, moov_src, out_dir / f"{rec_id}_2reencode.mp4", s, k, a)

    if "virtual" in variants:
        log = [json.loads(l) for l in (folder / "frames.jsonl").read_text().splitlines() if '"ts_ns"' in l]
        lts = np.array([r["ts_ns"] for r in log], dtype=np.int64)
        meta = json.loads((folder / "meta.json").read_text())
        W, H = meta["sensor_intrinsics"]["width"], meta["sensor_intrinsics"]["height"]
        rng = np.random.default_rng(0)

        def load(j, noise=False):
            im = Image.open(folder / "frames" / log[j]["file"]).convert("RGB")
            if im.size != (H, W):
                im = im.resize((H, W), Image.LANCZOS)
            # portrait render -> sensor orientation of the ARCore video (= CPU JPEG rotated 180 deg)
            a = np.rot90(np.array(im), 1)
            if noise:
                # a real phone sensor adds a little temporal noise (sigma ~1.5 of 255); a render has none
                a = np.clip(a + rng.normal(0, 1.5, a.shape), 0, 255).astype(np.uint8)
            return np.ascontiguousarray(a)

        import cv2
        idx0 = np.abs(lts[None, :] - vts[:, None]).argmin(1)
        small = lambda rgb: cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (320, 240)).astype(np.float32)
        with av.open(str(folder / "rec.mp4")) as c:
            real_flow = global_flow([small(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)])
        virt_flow = global_flow([small(load(j)) for j in idx0])
        lag, corr = estimate_lag(real_flow, virt_flow)
        # video frame k shows the virtual render logged with the timestamp of video frame k + lag
        target = vts[np.clip(np.arange(len(vts)) + lag, 0, len(vts) - 1)]
        idx = np.abs(lts[None, :] - target[:, None]).argmin(1)
        dt_ms = np.abs(lts[idx] - target) / 1e6
        frames = [load(j, noise=True) for j in idx]
        report.update(lag_frames=int(lag), lag_motion_corr=round(float(corr), 3))
        s, k, a = encode(frames, W, H)
        write_dataset(data, moov_src, out_dir / f"{rec_id}_3virtual.mp4", s, k, a)
        report.update(exact_frames=int((dt_ms < 12).sum()), nearest_frames=int((dt_ms >= 12).sum()),
                      max_dt_ms=round(float(dt_ms.max()), 1))
    print(json.dumps(report, indent=1))
    return report


def push(rec_id):
    for f in sorted((ROOT / "data" / "sim" / "playback").glob(f"{rec_id}_*.mp4")):
        adb("push", str(f), f"{PHONE_SIM}/playback/{f.name}")
        print("pushed", f.name)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pull"
    if cmd == "pull":
        print(pull())
    elif cmd == "probe":
        print(json.dumps(probe(sys.argv[2]), indent=1, ensure_ascii=False, default=str))
    elif cmd == "make":
        make(sys.argv[2], tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("copy", "reencode", "virtual"))
    elif cmd == "push":
        push(sys.argv[2])
