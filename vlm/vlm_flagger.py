#!/usr/bin/env python3
"""
PaveWatch - VLM flagger (Qwen3-VL via vLLM on the GB10)

Sends road images to the local VLM, classifies each one, flags real potholes,
and writes results the agent can read.

Usage:
  python vlm_flagger.py --images data/kaggle --limit 15          # quick test on 15 images
  python vlm_flagger.py --images data/kaggle                     # full run on a folder
  python vlm_flagger.py --csv images.csv --image-col frame_file  # use teammate's CSV (keeps its columns)

Outputs (in --out, default ./results):
  results.csv   one row per image with the VLM verdict
  flagged.json  only the flagged potholes -> input for the agent
"""
import argparse, base64, csv, io, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

LABELS = ["pothole", "speed_bump", "manhole", "rail_crossing", "shadow", "normal_road", "unclear"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PROMPT = """You are a road inspection assistant. Look at this image of a road surface.
Classify the most prominent road feature as exactly one of:
pothole, speed_bump, manhole, rail_crossing, shadow, normal_road, unclear.

Definitions:
- pothole: a hole or broken depression in the pavement surface
- speed_bump: an intentional raised strip across the road
- manhole: a round or square metal cover set into the road
- rail_crossing: railway tracks crossing the road
- shadow: a dark area that is only a shadow, not damage
- normal_road: intact pavement with no notable feature
- unclear: too blurry, dark, or obstructed to tell

If the label is pothole, estimate its size as small, medium, or large; otherwise size is null.
Respond with ONLY a JSON object and no other text:
{"label": "<one label>", "confidence": <0.0 to 1.0>, "size": "<small|medium|large or null>", "reason": "<under 15 words>"}"""


def encode_image(path, max_side=1024):
    """Return a base64 data URL. Downscales big images if Pillow is installed (faster, less memory)."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        data, mime = buf.getvalue(), "image/jpeg"
    except ImportError:
        data = Path(path).read_bytes()
        mime = "image/png" if Path(path).suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def parse_reply(text):
    """Pull the JSON object out of the model reply and sanity-check it."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # Qwen3 Thinking models reason first
    match = re.search(r"\{.*\}", text, re.DOTALL)
    out = json.loads(match.group(0)) if match else {}
    label = str(out.get("label", "unclear")).strip().lower().replace(" ", "_")
    try:
        conf = max(0.0, min(1.0, float(out.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0.0
    size = out.get("size") if label == "pothole" and out.get("size") in ("small", "medium", "large") else None
    return {"label": label if label in LABELS else "unclear", "confidence": conf,
            "size": size, "reason": str(out.get("reason", ""))[:120]}


def get_model(base_url):
    return requests.get(f"{base_url}/v1/models", timeout=10).json()["data"][0]["id"]


def classify(path, base_url, model):
    start = time.time()
    body = {"model": model, "temperature": 0, "max_tokens": 150,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encode_image(path)}},
                {"type": "text", "text": PROMPT}]}]}
    try:
        r = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=120)
        r.raise_for_status()
        result = parse_reply(r.json()["choices"][0]["message"]["content"])
        result["error"] = ""
    except Exception as e:  # never crash the batch on one bad image
        result = {"label": "unclear", "confidence": 0.0, "size": None, "reason": "", "error": str(e)[:200]}
    result["latency_s"] = round(time.time() - start, 2)
    return result


def truth_from_folder(folder):
    """Kaggle datasets usually sort images into folders like 'potholes' / 'normal'. Use that as ground truth."""
    name = folder.lower()
    if any(k in name for k in ["normal", "plain", "no_pothole", "nopothole", "non"]):
        return False
    return True if "pothole" in name else None


def collect(args):
    rows = []
    if args.csv:
        with open(args.csv, newline="") as f:
            reader = csv.DictReader(f)
            if args.image_col not in (reader.fieldnames or []):
                raise SystemExit(f"{args.csv} has no '{args.image_col}' column (has: {reader.fieldnames}). "
                                 "Pass --image-col, or point --images at a folder of frames.")
            for row in reader:
                p = Path(row[args.image_col])
                if not p.is_absolute() and args.images:
                    p = Path(args.images) / p
                rows.append({"image": str(p), **{k: v for k, v in row.items() if k != args.image_col}})
    else:
        for p in sorted(Path(args.images).rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({"image": str(p), "true_folder": p.parent.name})
    return rows[: args.limit] if args.limit else rows


def main():
    ap = argparse.ArgumentParser(description="PaveWatch VLM flagger")
    ap.add_argument("--images", help="folder of images (searched recursively)")
    ap.add_argument("--csv", help="optional CSV listing images (e.g. images.csv)")
    ap.add_argument("--image-col", default="frame_file", help="CSV column holding the image path")
    ap.add_argument("--url", default=os.environ.get("VLM_URL", "http://172.20.65.117:8001"),
                    help="vLLM server for Qwen3-VL (or set VLM_URL)")
    ap.add_argument("--threshold", type=float, default=0.6, help="min confidence to flag a pothole")
    ap.add_argument("--workers", type=int, default=2, help="parallel requests (keep 1-4 on the GB10)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images")
    ap.add_argument("--out", default="results", help="output folder")
    args = ap.parse_args()
    if not args.images and not args.csv:
        ap.error("give --images and/or --csv")

    rows = collect(args)
    try:
        model = get_model(args.url)
    except requests.RequestException as e:
        raise SystemExit(f"Can't reach vLLM at {args.url} ({type(e).__name__}). Is it running and listening "
                         "on 0.0.0.0, not just localhost? Use --url or VLM_URL to point elsewhere.")
    print(f"Model: {model} | images: {len(rows)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda r: classify(r["image"], args.url, model), rows))

    for i, (row, res) in enumerate(zip(rows, results), 1):
        row.update(res)
        row["flagged"] = res["label"] == "pothole" and res["confidence"] >= args.threshold
        print(f"[{i}/{len(rows)}] {Path(row['image']).name}: {res['label']} "
              f"({res['confidence']:.2f}){' FLAG' if row['flagged'] else ''} {res['error']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    flagged = [r for r in rows if r["flagged"]]
    (out / "flagged.json").write_text(json.dumps(flagged, indent=2))

    # Summary
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"\nLabels: {counts}\nFlagged: {len(flagged)} / {len(rows)}"
          f" | avg latency {sum(r['latency_s'] for r in rows) / max(len(rows), 1):.2f}s")

    # Accuracy check if Kaggle folder names give ground truth
    scored = [(truth_from_folder(r.get("true_folder", "")), r["flagged"]) for r in rows]
    scored = [(t, p) for t, p in scored if t is not None]
    if scored:
        tp = sum(t and p for t, p in scored); fp = sum((not t) and p for t, p in scored)
        fn = sum(t and not p for t, p in scored)
        acc = sum(t == p for t, p in scored) / len(scored)
        print(f"Accuracy {acc:.2%} | precision {tp / max(tp + fp, 1):.2%} | recall {tp / max(tp + fn, 1):.2%}"
              f" (on {len(scored)} images with folder labels)")
    print(f"Wrote {out / 'results.csv'} and {out / 'flagged.json'}")


if __name__ == "__main__":
    main()
