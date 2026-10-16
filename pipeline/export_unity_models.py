"""
Step 5 helper - copy a small test set of IKEA models into the Unity project.

Picks one well-measured model per test category (size error closest to 0) plus the
heaviest remaining model (for an FPS check), copies the .glb files to
unity-app/Assets/StreamingAssets/Models/ and writes models.json that the app reads
(name + expected size, so the app can show "expected vs measured").

Usage: python pipeline/export_unity_models.py [--categories dining_chair,sofa,...]
"""

import argparse
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# glTF-Transform CLI, installed locally: npm install --prefix tools/gltf @gltf-transform/cli
GLTF_TRANSFORM = ROOT / "tools" / "gltf" / "node_modules" / ".bin" / "gltf-transform.cmd"


def extensions_used(path: Path):
    b = path.read_bytes()
    n, = struct.unpack_from("<I", b, 12)
    return json.loads(b[20:20 + n]).get("extensionsUsed", [])


def make_unity_compatible(src: Path, dst: Path) -> str:
    """glTFast (Unity) cannot decode EXT_texture_webp textures - 5 of the first 7 test models
    failed on the phone with 'ExtensionUnsupported;EXT_texture_webp'. Re-encode WebP textures as
    PNG, then re-apply Draco (glTF-Transform decodes it while editing). Other files are copied."""
    ext = extensions_used(src)
    if "EXT_texture_webp" not in ext:
        shutil.copy2(src, dst)
        return "copied"
    with tempfile.TemporaryDirectory() as tmp:
        mid = Path(tmp) / "png.glb"
        subprocess.run([str(GLTF_TRANSFORM), "png", str(src), str(mid), "--formats", "webp"],
                       check=True, capture_output=True)
        if "KHR_draco_mesh_compression" in ext:
            subprocess.run([str(GLTF_TRANSFORM), "draco", str(mid), str(dst)], check=True, capture_output=True)
        else:
            shutil.copy2(mid, dst)
    if "EXT_texture_webp" in extensions_used(dst):
        raise RuntimeError(f"{dst.name}: WebP still present after conversion")
    return f"webp->png ({src.stat().st_size // 1024} KB -> {dst.stat().st_size // 1024} KB)"
DEFAULT_CATS = ["dining_chair", "sofa", "coffee_side_table", "bookcase_shelf", "floor_lamp", "rug"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default=",".join(DEFAULT_CATS))
    ap.add_argument("--out", default=str(ROOT / "unity-app" / "Assets" / "StreamingAssets" / "Models"))
    args = ap.parse_args()

    meta = [json.loads(l) for l in open(ROOT / "data" / "model_meta.jsonl", encoding="utf-8")]
    meta = [m for m in meta if m.get("ar_usable")]
    picks = []
    for cat in args.categories.split(","):
        cands = [m for m in meta if m["category"] == cat and m["status"] == "ok"]
        if cands:  # most accurate size first, then lighter file
            picks.append(min(cands, key=lambda m: (m["max_error_pct"], m["file_bytes"])))
    heaviest = max(meta, key=lambda m: m["triangles"])
    if heaviest not in picks:
        picks.append(heaviest)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = []
    for m in picks:
        src = ROOT / "data" / m["model_file"]
        dst = out / f"{m['item_no']}.glb"
        how = make_unity_compatible(src, dst)
        d = m["dims_cm"]
        models.append({"item_no": m["item_no"], "name": m["name"], "type_name": m["type_name"],
                       "category": m["category"], "file": f"Models/{dst.name}",
                       "w_cm": d["w"], "d_cm": d["d"], "h_cm": d["h"],
                       "triangles": m["triangles"], "file_kb": round(m["file_bytes"] / 1024)})
        print(f"{m['item_no']}  {m['category']:<18} {m['name']} {m['type_name']}  "
              f"{d['w']}x{d['d']}x{d['h']} cm  {m['triangles']:,} tris  err {m['max_error_pct']}%  [{how}]")
    (out / "models.json").write_text(json.dumps({"models": models}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(models)} models -> {out}")


if __name__ == "__main__":
    main()
