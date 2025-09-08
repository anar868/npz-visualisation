#!/usr/bin/env python3
# animate_kp2d_coords.py — animate 2D keypoints on a plain coordinate system (no video background)
# - Works with PosePipeline npz (object arrays, lists, (T,P,J,2/3) etc.)
# - Maps bml_movi_87 (56) -> COCO_25 -> OpenPose-25
# - Shows live animation; optionally saves MP4 if an output path is given

import sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

NPZ = sys.argv[1] if len(sys.argv) > 1 else "4_1.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else None   # e.g. "traj.mp4"
FPS = 30                                            # playback fps

# ---------- Names ----------
OPENPOSE_25 = [
    "Nose","Sternum","Right Shoulder","Right Elbow","Right Wrist","Left Shoulder",
    "Left Elbow","Left Wrist","Pelvis","Right Hip","Right Knee","Right Ankle",
    "Left Hip","Left Knee","Left Ankle","Right Eye","Left Eye","Right Ear","Left Ear",
    "Left Big Toe","Left Little Toe","Left Heel","Right Big Toe","Right Little Toe","Right Heel",
]
COCO_25 = [
    "Sternum","Nose","Pelvis","Left Shoulder","Left Elbow","Left Wrist","Left Hip","Left Knee",
    "Left Ankle","Right Shoulder","Right Elbow","Right Wrist","Right Hip","Right Knee","Right Ankle",
    "Left Eye","Left Ear","Right Eye","Right Ear","Left Big Toe","Left Little Toe","Left Heel",
    "Right Big Toe","Right Little Toe","Right Heel",
]
BML_MOVI_87 = [
    "backneck","upperback","clavicle","sternum","umbilicus","lfronthead","lbackhead","lback","lshom",
    "lupperarm","lelbm","lforearm","lwrithumbside","lwripinkieside","lfin","lasis","lpsis","lfrontthigh",
    "lthigh","lknem","lankm","Left Heel","lfifthmetatarsal","Left Big Toe","lcheek","lbreast","lelbinner",
    "lwaist","lthumb","lfrontinnerthigh","linnerknee","lshin","lfirstmetatarsal","lfourthtoe","lscapula",
    "lbum","rfronthead","rbackhead","rback","rshom","rupperarm","relbm","rforearm","rwrithumbside",
    "rwripinkieside","rfin","rasis","rpsis","rfrontthigh","rthigh","rknem","rankm","Right Heel",
    "rfifthmetatarsal","Right Big Toe","rcheek","rbreast","relbinner","rwaist","rthumb","rfrontinnerthigh",
    "rinnerknee","rshin","rfirstmetatarsal","rfourthtoe","rscapula","rbum","Head","mhip","Pelvis","Sternum",
    "Left Ankle","Left Elbow","Left Hip","Left Hand","Left Knee","Left Shoulder","Left Wrist","Left Foot",
    "Right Ankle","Right Elbow","Right Hip","Right Hand","Right Knee","Right Shoulder","Right Wrist","Right Foot",
]
BML56 = BML_MOVI_87[-56:]

COCO_TO_BML56 = {
    "Nose": "Head",
    "Left Eye": "lfronthead",
    "Right Eye": "rfronthead",
    "Left Ear": "lbackhead",
    "Right Ear": "rbackhead",
    "Left Little Toe": "lfourthtoe",
    "Right Little Toe": "rfourthtoe",
}

EDGES_25 = [
    (0,1),(0,15),(0,16),(15,17),(16,18),
    (1,2),(1,5),(1,8),(8,12),(8,9),
    (2,3),(3,4),
    (5,6),(6,7),
    (9,10),(10,11),(11,24),(11,22),(11,23),
    (12,13),(13,14),(14,21),(14,19),(14,20),
]

# ---------- helpers ----------
def unwrap_first(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
        if x.shape == (): return x.item()
        if x.shape[0] >= 1: return x[0]
    if isinstance(x, (list, tuple)):
        return x[0]
    return x

def to_T_P_J_C(kp):
    arr = np.asarray(kp)
    while arr.ndim > 4 and 1 in arr.shape[:-1]:
        arr = np.squeeze(arr)
    if arr.ndim == 3:  # (T,J,C)
        T,J,C = arr.shape
        if C not in (2,3): raise ValueError(f"C must be 2 or 3; got {C}")
        return arr.reshape(T,1,J,C)
    if arr.ndim == 4:  # (A,B,J,C): choose time as larger of A,B
        A,B,J,C = arr.shape
        if C not in (2,3): raise ValueError(f"C must be 2 or 3; got {C}")
        return arr if A >= B else np.swapaxes(arr,0,1)
    if arr.ndim == 5 and arr.shape[0] == 1:  # (1,T,J,C)
        arr = np.squeeze(arr, axis=0)
        return to_T_P_J_C(arr)
    raise ValueError(f"Unsupported kp2d shape: {arr.shape}")

def norm_name(s): return " ".join(s.lower().strip().split())

def build_index(source_names, wanted_names, alias_map=None):
    src_map = {norm_name(n): i for i, n in enumerate(source_names)}
    idx, missing = [], []
    for w in wanted_names:
        w_src = alias_map.get(w, w) if alias_map else w
        k = norm_name(w_src)
        if k not in src_map:
            missing.append((w, w_src)); idx.append(None)
        else:
            idx.append(src_map[k])
    if missing:
        lines = [f"  wanted '{w}' → tried '{ws}' (not in source)" for (w,ws) in missing]
        raise KeyError("Missing joints in source:\n" + "\n".join(lines))
    if len(set(idx)) != len(idx):
        raise ValueError("Non-unique indices in mapping")
    return idx

def map_to_openpose25(kp_TPJ2or3):
    T,P,J,C = kp_TPJ2or3.shape
    if J == 25:
        idx_op = [COCO_25.index(n) for n in OPENPOSE_25]
        return kp_TPJ2or3[:, :, idx_op, :]
    if J == 56:
        idx_coco = build_index(BML56, COCO_25, alias_map=COCO_TO_BML56)
        kp25c = kp_TPJ2or3[:, :, idx_coco, :]
        idx_op = [COCO_25.index(n) for n in OPENPOSE_25]
        return kp25c[:, :, idx_op, :]
    raise ValueError(f"Unsupported joint count J={J}; expected 25 or 56.")

# ---------- load ----------
data = np.load(NPZ, allow_pickle=True)
key_name = "keypoints2d" if "keypoints2d" in data.files else ("keypoints" if "keypoints" in data.files else None)
if key_name is None:
    raise KeyError(f"No 'keypoints2d' or 'keypoints' in {NPZ}. Keys: {list(data.files)}")

kp2d_raw = unwrap_first(data[key_name])
kp_TPJ = to_T_P_J_C(kp2d_raw)        # (T,P,J,C)
T,P,J,C = kp_TPJ.shape

use_edges = J in (25, 56)
if use_edges:
    kp_TPJ = map_to_openpose25(kp_TPJ)
    J = 25

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect("equal", adjustable="box")
# ax.invert_yaxis()  # image coord convention
scatters = []
lines = []

# Fix axis limits to full data extents so playback doesn't autoscale
XY = kp_TPJ[..., :2].reshape(-1, 2)
xmn, ymn = np.nanmin(XY, axis=0)
xmx, ymx = np.nanmax(XY, axis=0)
pad_x = 0.05 * (xmx - xmn + 1e-6)
pad_y = 0.05 * (ymx - ymn + 1e-6)
ax.set_xlim(xmn - pad_x, xmx + pad_x)
ax.set_ylim(ymn - pad_y, ymx + pad_y)
title = ax.set_title(f"2D skeleton — frame 0/{T-1}")
# Flip Y axis to make it Cartesian (Y up)
H = np.nanmax(kp_TPJ[...,1])   # assume image height is max Y
kp_TPJ[...,1] = H - kp_TPJ[...,1]

# initialize artists
colors = ["C0","C1","C2","C3","C4","C5","C6","C7"]
for p in range(P):
    pts = kp_TPJ[0, p, :, :2]
    s = ax.scatter(pts[:,0], pts[:,1], s=25, color=colors[p % len(colors)], alpha=0.9)
    scatters.append(s)
    if use_edges:
        # one Line2D per edge per person
        person_lines = []
        for (i,j) in EDGES_25:
            ln, = ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]],
                          lw=2, color=colors[p % len(colors)], alpha=0.8)
            person_lines.append(ln)
        lines.append(person_lines)
    else:
        lines.append([])

def update(frame_idx):
    title.set_text(f"2D skeleton — frame {frame_idx}/{T-1}")
    for p in range(P):
        pts = kp_TPJ[frame_idx, p, :, :2]
        scatters[p].set_offsets(pts)
        if use_edges:
            for ln, (i,j) in zip(lines[p], EDGES_25):
                ln.set_data([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]])
    return [title] + scatters + [ln for sub in lines for ln in sub]

anim = FuncAnimation(fig, update, frames=T, interval=1000/FPS, blit=False)

if OUT:
    try:
        writer = FFMpegWriter(fps=FPS, bitrate=4000)
        anim.save(OUT, writer=writer)
        print(f"Saved animation to {OUT}")
    except Exception as e:
        print("FFmpeg writer failed; showing interactively instead. Error:", e)
        plt.show()
else:
    plt.show()