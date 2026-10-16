"""
Minimal ISO-BMFF (MP4) box tree: parse, edit, write. Enough to swap the video samples of an ARCore
dataset while every other byte (IMU + metadata tracks, uuid boxes, timing) stays as ARCore wrote it.
"""

import struct

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta", b"mvex"}


class Box:
    def __init__(self, typ, payload=b"", children=None):
        self.type, self.payload, self.children = typ, payload, children

    def find(self, typ):
        return next((c for c in self.children or [] if c.type == typ), None)

    def findall(self, typ):
        return [c for c in self.children or [] if c.type == typ]

    def serialize(self):
        body = b"".join(c.serialize() for c in self.children) if self.children is not None else self.payload
        size = 8 + len(body)
        if size > 0xFFFFFFFF:
            return struct.pack(">I4sQ", 1, self.type, size + 8) + body
        return struct.pack(">I4s", size, self.type) + body


def parse(data, off=0, end=None):
    end = len(data) if end is None else end
    out = []
    while off + 8 <= end:
        size, typ = struct.unpack_from(">I4s", data, off)
        hdr = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, off + 8)[0]
            hdr = 16
        elif size == 0:
            size = end - off
        if typ in CONTAINERS:
            out.append(Box(typ, children=parse(data, off + hdr, off + size)))
        else:
            out.append(Box(typ, payload=bytes(data[off + hdr:off + size])))
        out[-1].offset, out[-1].header = off, hdr
        off += size
    return out


# ---------------------------------------------------------------- sample tables
def _full(payload):
    return payload[:4], payload[4:]


def read_stsz(p):
    _, b = _full(p)
    fixed, n = struct.unpack_from(">II", b)
    return [fixed] * n if fixed else list(struct.unpack_from(f">{n}I", b, 8))


def read_stco(p, wide=False):
    _, b = _full(p)
    n, = struct.unpack_from(">I", b)
    return list(struct.unpack_from(f">{n}{'Q' if wide else 'I'}", b, 4))


def read_stsc(p):
    _, b = _full(p)
    n, = struct.unpack_from(">I", b)
    return [struct.unpack_from(">III", b, 4 + 12 * i) for i in range(n)]


def read_stts(p):
    _, b = _full(p)
    n, = struct.unpack_from(">I", b)
    return [struct.unpack_from(">II", b, 4 + 8 * i) for i in range(n)]


def sample_offsets(stbl):
    """Absolute file offset + size of every sample of a track."""
    sizes = read_stsz(stbl.find(b"stsz").payload)
    co = stbl.find(b"stco")
    chunks = read_stco(co.payload) if co else read_stco(stbl.find(b"co64").payload, wide=True)
    stsc = read_stsc(stbl.find(b"stsc").payload)
    out, s = [], 0
    for ci, base in enumerate(chunks, start=1):
        per = next(spc for first, spc, _ in reversed(stsc) if first <= ci)
        off = base
        for _ in range(per):
            if s >= len(sizes):
                break
            out.append((off, sizes[s]))
            off += sizes[s]
            s += 1
    return out


def sample_times(stbl):
    """Decode timestamps (track timescale ticks) of every sample."""
    t, out = 0, []
    for count, delta in read_stts(stbl.find(b"stts").payload):
        for _ in range(count):
            out.append(t)
            t += delta
    return out


def tracks(moov):
    """[(handler, mime/codec, trak box)]"""
    res = []
    for trak in moov.findall(b"trak"):
        mdia = trak.find(b"mdia")
        handler = mdia.find(b"hdlr").payload[8:12].decode()
        stsd = mdia.find(b"minf").find(b"stbl").find(b"stsd").payload
        entry = stsd[8 + 4:8 + 8].decode(errors="replace")
        if entry == "mett":
            mime = stsd[8 + 8 + 8:].split(b"\0")[1].decode(errors="replace") if b"\0" in stsd[16:] else ""
            mime = stsd[16 + 8:].strip(b"\0").split(b"\0")[0].decode(errors="replace")
            entry = mime
        res.append((handler, entry, trak))
    return res


def stbl_of(trak):
    return trak.find(b"mdia").find(b"minf").find(b"stbl")
