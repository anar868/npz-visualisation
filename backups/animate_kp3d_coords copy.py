#!/usr/bin/env python3
# animate_kp3d_coords.py — animate 3D keypoints on axes (no video)
# - Handles PosePipeline npz (lists/object arrays ok)
# - Normalizes to (T,P,J,3)
# - Maps bml_movi_87 (56) -> COCO_25 -> OpenPose-25 for edges
# - Centers at pelvis, optional axis reordering/flips, mm→m scaling
# - Saves MP4 if an output path is given

import sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa

NPZ = sys.argv[1] if len(sys.argv) > 1 else "4_1.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else None
FPS = 30

# ---- quick knobs ----
AXIS_ORDER = (0, 2, 1)    # keep as (x,y,z)
FLIPS      = (+1, +1, -1) # flip Z instead
SCALE_MM   = True
CENTER_AT  = "Pelvis"

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
    "Nose":"Head","Left Eye":"lfronthead","Right Eye":"rfronthead",
    "Left Ear":"lbackhead","Right Ear":"rbackhead",
    "Left Little Toe":"lfourthtoe","Right Little Toe":"rfourthtoe",
}
EDGES_25 = [
    (0,1),(0,15),(0,16),(15,17),(16,18),
    (1,2),(1,5),(1,8),(8,12),(8,9),
    (2,3),(3,4),(5,6),(6,7),
    (9,10),(10,11),(11,24),(11,22),(11,23),
    (12,13),(13,14),(14,21),(14,19),(14,20),
]

# ----- helpers -----
def unwrap_first(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
        if x.shape == (): return x.item()
        if x.shape[0] >= 1: return x[0]
    if isinstance(x, (list, tuple)): return x[0]
    return x

def to_T_P_J_3(kp):
    arr = np.asarray(kp)
    # move coords last
    coord_axes = [i for i,s in enumerate(arr.shape) if s == 3]
    if coord_axes and coord_axes[-1] != arr.ndim-1:
        arr = np.moveaxis(arr, coord_axes[-1], -1)
    # squeeze extra singletons
    while arr.ndim > 4 and 1 in arr.shape[:-1]:
        arr = np.squeeze(arr)
    if arr.ndim == 3:
        A,B,C = arr.shape
        if C != 3: raise ValueError(f"Last dim must be 3; got {C}")
        out = arr if A >= B else np.swapaxes(arr,0,1)  # frames >> joints
        T,J,_ = out.shape
        return out.reshape(T,1,J,3)
    if arr.ndim == 4:
        A,B,J,C = arr.shape
        if C != 3: raise ValueError(f"Last dim must be 3; got {C}")
        return arr if A >= B else np.swapaxes(arr,0,1)  # choose time as larger axis
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
        return to_T_P_J_3(arr)
    raise ValueError(f"Unsupported kp3d shape: {arr.shape}")

def norm_name(s): return " ".join(s.lower().strip().split())

def build_index(source_names, wanted_names, alias_map=None):
    src = {norm_name(n):i for i,n in enumerate(source_names)}
    idx, missing = [], []
    for w in wanted_names:
        wsrc = alias_map.get(w,w) if alias_map else w
        k = norm_name(wsrc)
        if k not in src:
            missing.append((w,wsrc)); idx.append(None)
        else:
            idx.append(src[k])
    if missing:
        msg = "\n".join([f"  wanted '{w}' → tried '{ws}' (not in source)" for w,ws in missing])
        raise KeyError("Missing joints in source:\n"+msg)
    if len(set(idx)) != len(idx):
        raise ValueError("Non-unique indices in mapping")
    return idx

def to_openpose25_TPJ(kp_TPJ3):
    T,P,J,C = kp_TPJ3.shape
    if J == 25:
        idx_op = [COCO_25.index(n) for n in OPENPOSE_25]
        return kp_TPJ3[:, :, idx_op, :]
    if J == 56:
        idx_coco = build_index(BML56, COCO_25, alias_map=COCO_TO_BML56)
        kp25c = kp_TPJ3[:, :, idx_coco, :]
        idx_op = [COCO_25.index(n) for n in OPENPOSE_25]
        return kp25c[:, :, idx_op, :]
    raise ValueError(f"Unsupported J={J}; expected 25 or 56.")

def center_points(pts, names, center_at="Pelvis"):
    name_to_idx = {norm_name(n):i for i,n in enumerate(names)}
    if center_at.lower() == "none": return pts
    for cand in [center_at,"mhip","Pelvis","Sternum"]:
        j = name_to_idx.get(norm_name(cand))
        if j is not None and 0 <= j < len(pts):
            return pts - pts[j]
    return pts - np.median(pts, axis=0)

def set_axes_equal(ax, pts):
    mn = pts.min(axis=0).astype(float)
    mx = pts.max(axis=0).astype(float)
    ctr = (mn+mx)/2.0
    r = np.max(mx-mn)/2.0
    ax.set_xlim(ctr[0]-r, ctr[0]+r)
    ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)

# ----- load -----
data = np.load(NPZ, allow_pickle=True)
key = "keypoints3d" if "keypoints3d" in data.files else ("keypoints" if "keypoints" in data.files else None)
if key is None:
    raise KeyError(f"No 'keypoints3d' or 'keypoints' in {NPZ}. Keys: {list(data.files)}")

kp3d_raw = unwrap_first(data[key])
kp_TPJ3  = to_T_P_J_3(kp3d_raw)  # (T,P,J,3)
T,P,J,C  = kp_TPJ3.shape

# Map to OpenPose 25 if possible (for edges)
if J in (25,56):
    kp_TPJ3 = to_openpose25_TPJ(kp_TPJ3)
    names = OPENPOSE_25
    edges = EDGES_25
    J = 25
else:
    names = [f"J{id}"]*J
    edges = []

# Axis order / flips / scale / centering (applied to entire sequence)
pts_all = kp_TPJ3[..., AXIS_ORDER] * np.array(FLIPS)[None,None,None,:]
if SCALE_MM:
    pts_all *= 0.001
if CENTER_AT.lower() != "none":
    pelvis_idx = 8 if names is OPENPOSE_25 else 0
    center = pts_all[:,:,pelvis_idx:pelvis_idx+1,:]  # (T,P,1,3)
    pts_all = pts_all - center

# ----- figure / artists -----
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection="3d")
colors = ["C0","C1","C2","C3","C4","C5","C6","C7"]

# overall bounds fixed for the whole animation
xyz = pts_all.reshape(-1,3)
mn = np.nanmin(xyz, axis=0); mx = np.nanmax(xyz, axis=0)
ctr = (mn+mx)/2.0
r = np.max(mx-mn)/2.0
ax.set_xlim(ctr[0]-r, ctr[0]+r)
ax.set_ylim(ctr[1]-r, ctr[1]+r)
ax.set_zlim(ctr[2]-r, ctr[2]+r)
try: ax.set_box_aspect([1,1,1])
except Exception: pass
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
title = ax.set_title(f"3D skeleton — frame 0/{T-1}")

scatters = []
lines = []
t0 = 0
for p in range(P):
    pts = pts_all[t0, p]            # (J,3)
    s = ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=25, color=colors[p%len(colors)], alpha=0.9)
    scatters.append(s)
    if edges:
        person_lines = []
        for (i,j) in edges:
            ln, = ax.plot([pts[i,0], pts[j,0]],
                          [pts[i,1], pts[j,1]],
                          [pts[i,2], pts[j,2]],
                          lw=2, color=colors[p%len(colors)], alpha=0.85)
            person_lines.append(ln)
        lines.append(person_lines)
    else:
        lines.append([])

def update(t):
    title.set_text(f"3D skeleton — frame {t}/{T-1}")
    for p in range(P):
        pts = pts_all[t, p]
        scatters[p]._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
        if edges:
            for ln, (i,j) in zip(lines[p], edges):
                ln.set_data_3d([pts[i,0], pts[j,0]],
                               [pts[i,1], pts[j,1]],
                               [pts[i,2], pts[j,2]])
    return [title] + scatters + [ln for sub in lines for ln in sub]

anim = FuncAnimation(fig, update, frames=T, interval=1000/FPS, blit=False)

if OUT:
    try:
        writer = FFMpegWriter(fps=FPS, bitrate=6000)
        anim.save(OUT, writer=writer)
        print(f"Saved animation to {OUT}")
    except Exception as e:
        print("FFmpeg writer failed; showing interactively instead. Error:", e)
        plt.show()
else:
    plt.show()