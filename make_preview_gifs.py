"""Build one animated GIF per object from the first sequence of each.

Walks a rendered dataset directory, picks the lowest-numbered sequence for every
object, and writes an animated GIF of its frames to a target directory.  Useful
for eyeballing a whole render run at a glance.

Layout written by generate_traversals.py:

    <root>/seqs/<synset_id>/<obj_id[:2]>/<obj_id>/seq_%04d/frame_%04d.jpg

The lowest sequence index is NOT always seq_0000: in single-factor mode seq_idx
IS the factor index, so an object restricted via --factors may start at e.g.
seq_0002.  We therefore take min() of what is present rather than assuming.

Pure Python + Pillow — no Blender, no BlenderProc.

    python make_preview_gifs.py --dataset-dir DATASET --out-dir gifs/
"""
import argparse
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool

from PIL import Image

_SEQ_RE = re.compile(r'seq_(\d+)$')
_FRAME_RE = re.compile(r'frame_(\d+)\.jpg$', re.IGNORECASE)


def _seq_index(path):
    """Parse the trailing seq_NNNN index, or None if it doesn't match."""
    m = _SEQ_RE.search(os.path.basename(path.rstrip(os.sep)))
    return int(m.group(1)) if m else None


def find_sequence_dirs(root):
    """Return every seq_* directory under `root`.

    Tries the canonical <root>/seqs/<synset>/<xx>/<obj>/seq_* depth first, then
    falls back to a full recursive walk so the script also works when pointed
    straight at a subtree (e.g. a bare directory of seq_* dirs).
    """
    found = []
    seqs_root = os.path.join(root, 'seqs')
    if os.path.isdir(seqs_root):
        for synset in sorted(os.listdir(seqs_root)):
            p1 = os.path.join(seqs_root, synset)
            if not os.path.isdir(p1):
                continue
            for prefix in sorted(os.listdir(p1)):
                p2 = os.path.join(p1, prefix)
                if not os.path.isdir(p2):
                    continue
                for obj_id in sorted(os.listdir(p2)):
                    p3 = os.path.join(p2, obj_id)
                    if not os.path.isdir(p3):
                        continue
                    for seq in sorted(os.listdir(p3)):
                        p4 = os.path.join(p3, seq)
                        if os.path.isdir(p4) and _seq_index(p4) is not None:
                            found.append(p4)
    if not found:
        for dirpath, dirnames, _files in os.walk(root):
            for d in sorted(dirnames):
                p = os.path.join(dirpath, d)
                if _seq_index(p) is not None:
                    found.append(p)
    return sorted(found)


def first_sequence_per_object(seq_dirs):
    """Map each object directory to its lowest-numbered sequence directory."""
    best = {}
    for seq_dir in seq_dirs:
        obj_dir = os.path.dirname(seq_dir.rstrip(os.sep))
        idx = _seq_index(seq_dir)
        if idx is None:
            continue
        current = best.get(obj_dir)
        if current is None or idx < current[0]:
            best[obj_dir] = (idx, seq_dir)
    return {obj: seq for obj, (_idx, seq) in best.items()}


def frame_paths(seq_dir):
    """Frame files in numeric order (not lexical, so padding changes are safe)."""
    entries = []
    for name in os.listdir(seq_dir):
        m = _FRAME_RE.search(name)
        if m:
            entries.append((int(m.group(1)), os.path.join(seq_dir, name)))
    return [p for _i, p in sorted(entries)]


def gif_name(obj_dir):
    """'<synset>_<obj_id>.gif' from .../<synset>/<xx>/<obj_id>, else '<obj_id>.gif'."""
    obj_id = os.path.basename(obj_dir.rstrip(os.sep))
    parts = obj_dir.rstrip(os.sep).split(os.sep)
    # parts[-1]=obj_id, parts[-2]=obj_id[:2] prefix, parts[-3]=synset
    if len(parts) >= 3 and parts[-2] == obj_id[:2]:
        return f"{parts[-3]}_{obj_id}.gif"
    return f"{obj_id}.gif"


def write_gif(frames, out_path, fps=20, loop=0, resize=None):
    """Encode `frames` (file paths) into an animated GIF at out_path.

    GIF allows 256 colours, so the frames must be quantised. The palette is
    derived from ALL frames stacked together, not just the first: these
    traversals sweep floor_hue / spot_hue across a sequence, so hues that only
    appear later are absent from frame 0. Quantising against a frame-0 palette
    would render those later hues wrong and — when the mapped frames collapse to
    identical images — Pillow's optimiser merges them, producing a single-frame
    "animation" with the durations summed.

    Sharing one palette across every frame also keeps colours from shifting
    frame to frame, which would read as flicker on the smooth floor gradient.

    Returns the number of frames submitted. Pillow may still merge genuinely
    identical consecutive frames, so a static sequence can yield fewer.
    """
    if not frames:
        raise ValueError("no frames to write")
    images = []
    for p in frames:
        img = Image.open(p).convert('RGB')
        if resize:
            img = img.resize((resize, resize), Image.NEAREST)
        images.append(img)

    # Stack every frame into one tall image and quantise that, so the palette
    # covers colours from the whole sequence.
    w, h = images[0].size
    stacked = Image.new('RGB', (w, h * len(images)))
    for i, im in enumerate(images):
        stacked.paste(im, (0, i * h))
    palette_img = stacked.quantize(colors=256, method=Image.MEDIANCUT)
    quantized = [im.quantize(palette=palette_img) for im in images]

    duration_ms = max(int(round(1000.0 / fps)), 1)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # disposal=1 ("do not dispose") leaves the previous frame in place. Do NOT
    # use disposal=2 ("restore to background"): every frame here is fully opaque,
    # so there is nothing to dispose, and viewers implement "background" as
    # transparent — which shows up as a grey/white checkerboard flashing between
    # frames. It also makes any partial frame the optimiser emits composite onto
    # a cleared canvas instead of the previous image.
    quantized[0].save(
        out_path, save_all=True, append_images=quantized[1:],
        duration=duration_ms, loop=loop, optimize=True, disposal=1)
    return len(quantized)


def _job(task):
    """Worker: (obj_dir, seq_dir, out_path, fps, loop, resize, modal_n)."""
    obj_dir, seq_dir, out_path, fps, loop, resize, modal_n = task
    frames = frame_paths(seq_dir)
    if not frames:
        return (obj_dir, None, "no frames")
    note = ""
    if modal_n and len(frames) < modal_n:
        note = f"incomplete ({len(frames)}/{modal_n} frames)"
    try:
        n = write_gif(frames, out_path, fps=fps, loop=loop, resize=resize)
    except Exception as e:  # keep one bad sequence from killing the batch
        return (obj_dir, None, f"ERROR: {e}")
    return (obj_dir, n, note)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-dir', required=True,
                        help="Rendered dataset root (containing seqs/), or any "
                             "directory tree containing seq_* dirs")
    parser.add_argument('--out-dir', required=True,
                        help="Target directory for the .gif files")
    parser.add_argument('--fps', type=float, default=16.0,
                        help="Playback frames per second (default: 20)")
    parser.add_argument('--loop', type=int, default=0,
                        help="GIF loop count; 0 = forever (default: 0)")
    parser.add_argument('--resize', type=int, default=None, metavar='PX',
                        help="Resize frames to PXxPX (nearest-neighbour). Default: "
                             "native render resolution.")
    parser.add_argument('--limit', type=int, default=None,
                        help="Only process the first N objects (quick sample)")
    parser.add_argument('--jobs', type=int, default=1,
                        help="Parallel worker processes (default: 1)")
    parser.add_argument('--overwrite', action='store_true',
                        help="Re-encode GIFs that already exist (default: skip)")
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    def log(*a, **kw):
        if not args.quiet:
            print(*a, **kw, flush=True)

    if not os.path.isdir(args.dataset_dir):
        sys.exit(f"ERROR: dataset dir not found: {args.dataset_dir}")

    seq_dirs = find_sequence_dirs(args.dataset_dir)
    if not seq_dirs:
        sys.exit(f"ERROR: no seq_* directories found under {args.dataset_dir}")
    firsts = first_sequence_per_object(seq_dirs)
    log(f"Found {len(seq_dirs)} sequences across {len(firsts)} objects in "
        f"{args.dataset_dir}")

    # Modal frame count, to spot truncated sequences.
    counts = Counter(len(frame_paths(s)) for s in firsts.values())
    modal_n = counts.most_common(1)[0][0] if counts else 0

    objs = sorted(firsts)
    if args.limit is not None:
        objs = objs[:args.limit]
        log(f"Limited to {len(objs)} objects (--limit)")

    tasks, skipped = [], 0
    for obj_dir in objs:
        out_path = os.path.join(args.out_dir, gif_name(obj_dir))
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue
        tasks.append((obj_dir, firsts[obj_dir], out_path,
                      args.fps, args.loop, args.resize, modal_n))
    if skipped:
        log(f"Skipping {skipped} objects with an existing GIF (--overwrite to redo)")

    os.makedirs(args.out_dir, exist_ok=True)
    written = failed = 0
    if args.jobs > 1 and tasks:
        with Pool(args.jobs) as pool:
            results = pool.imap_unordered(_job, tasks)
            for i, (obj_dir, n, note) in enumerate(results, 1):
                if n is None:
                    failed += 1
                    log(f"[{i}/{len(tasks)}] {os.path.basename(obj_dir)} — {note}")
                else:
                    written += 1
                    log(f"[{i}/{len(tasks)}] {os.path.basename(obj_dir)} — "
                        f"{n} frames {note}".rstrip())
    else:
        for i, task in enumerate(tasks, 1):
            obj_dir, n, note = _job(task)
            if n is None:
                failed += 1
                log(f"[{i}/{len(tasks)}] {os.path.basename(obj_dir)} — {note}")
            else:
                written += 1
                log(f"[{i}/{len(tasks)}] {os.path.basename(obj_dir)} — "
                    f"{n} frames {note}".rstrip())

    log(f"\nDone. {written} GIFs written to {args.out_dir}"
        + (f", {skipped} skipped" if skipped else "")
        + (f", {failed} failed" if failed else ""))


if __name__ == '__main__':
    main()
