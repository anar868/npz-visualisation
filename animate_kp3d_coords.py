#!/usr/bin/env python3
# animate_kp3d_coords.py — animate 3D keypoints on axes (no video)
# Handles PosePipeline npz (lists/object arrays ok)
# - Normalizes to (T,P,J,3)
# - Filters 580 → BML-MoVi-87, or maps 56→COCO_25→OpenPose-25
# - Centers at pelvis, optional axis reordering/flips, mm→m scaling
# - Saves MP4 if an output path is given

import sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

NPZ = sys.argv[1] if len(sys.argv) > 1 else "4_1.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else None
FPS = 30

AXIS_ORDER = (0, 2, 1)    
FLIPS      = (+1, +1, -1) 
SCALE_MM   = True
CENTER_AT  = "pelv"

BML87_IDX = np.array([
    264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,
    284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,
    304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,
    324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,
    344,345,346,347,348,349,350
], dtype=int)

BML87_EDGES = np.array([
    [67, 70],[68, 69],[68, 73],[68, 81],[69, 70],[70, 76],[70, 84],[71, 75],[71, 78],
    [72, 76],[72, 77],[73, 75],[74, 77],[79, 83],[79, 86],[80, 84],[80, 85],[81, 83],[82, 85]
], dtype=int)

def unwrap_first(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
        if x.shape == (): return x.item()
        if x.shape[0] >= 1: return x[0]
    if isinstance(x, (list, tuple)): return x[0]
    return x

def to_T_P_J_3(kp):
    arr = np.asarray(kp)
    coord_axes = [i for i,s in enumerate(arr.shape) if s == 3]
    if coord_axes and coord_axes[-1] != arr.ndim-1:
        arr = np.moveaxis(arr, coord_axes[-1], -1)
    while arr.ndim > 4 and 1 in arr.shape[:-1]:
        arr = np.squeeze(arr)
    if arr.ndim == 3:
        A,B,C = arr.shape
        if C!=3: raise ValueError("Last dim must be 3")
        out = arr if A>=B else np.swapaxes(arr,0,1)
        T,J,_ = out.shape
        return out.reshape(T,1,J,3)
    if arr.ndim == 4:
        A,B,J,C = arr.shape
        return arr if A>=B else np.swapaxes(arr,0,1)
    if arr.ndim == 5 and arr.shape[0]==1:
        return to_T_P_J_3(np.squeeze(arr,0))
    raise ValueError(f"Unsupported shape {arr.shape}")

data = np.load(NPZ, allow_pickle=True)
key = "keypoints3d" if "keypoints3d" in data.files else ("keypoints" if "keypoints" in data.files else None)
if key is None:
    raise KeyError(f"No 'keypoints3d' or 'keypoints' in {NPZ}. Keys: {list(data.files)}")

kp3d_raw = unwrap_first(data[key])
kp_TPJ3  = to_T_P_J_3(kp3d_raw)  # (T,P,J,3)
T,P,J,C  = kp_TPJ3.shape

# filter to BML-MoVi-87
kp_TPJ3 = kp_TPJ3[:,:,BML87_IDX,:]   # (T,P,87,3)
names = ['backneck', 'upperback', 'clavicle', 'sternum', 'umbilicus',
       'lfronthead', 'lbackhead', 'lback', 'lshom', 'lupperarm',
       'lelbm', 'lforearm', 'lwrithumbside', 'lwripinkieside',
       'lfin', 'lasis', 'lpsis', 'lfrontthigh', 'lthigh', 'lknem',
       'lankm', 'lhee', 'lfifthmetatarsal', 'ltoe', 'lcheek',
       'lbreast', 'lelbinner', 'lwaist', 'lthum',
       'lfrontinnerthigh', 'linnerknee', 'lshin', 'lfirstmetatarsal',
       'lfourthtoe', 'lscapula', 'lbum', 'rfronthead', 'rbackhead',
       'rback', 'rshom', 'rupperarm', 'relbm', 'rforearm',
       'rwrithumbside', 'rwripinkieside', 'rfin', 'rasis', 'rpsis',
       'rfrontthigh', 'rthigh', 'rknem', 'rankm', 'rhee',
       'rfifthmetatarsal', 'rtoe', 'rcheek', 'rbreast', 'relbinner',
       'rwaist', 'rthum', 'rfrontinnerthigh', 'rinnerknee', 'rshin',
       'rfirstmetatarsal', 'rfourthtoe', 'rscapula', 'rbum', 'head',
       'mhip', 'pelv', 'thor', 'lank', 'lel', 'lhip', 'lhan',
       'lkne', 'lsho', 'lwri', 'lfoo', 'rank', 'rel', 'rhip',
       'rhan', 'rkne', 'rsho', 'rwri', 'rfoo']

edges = [tuple(e) for e in BML87_EDGES]

# Axis order / flips / scale / centering
pts_all = kp_TPJ3[..., AXIS_ORDER] * np.array(FLIPS)[None,None,None,:]
if SCALE_MM: pts_all *= 0.001
if CENTER_AT.lower() != "none" and len(names)>0:
    try:
        pelvis_idx = names.index(CENTER_AT)
        center = pts_all[:,:,pelvis_idx:pelvis_idx+1,:]
        pts_all = pts_all - center
    except Exception:
        pass

# ----- figure / artists -----
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection="3d")

xyz = pts_all.reshape(-1,3)
mn = np.nanmin(xyz,0); mx = np.nanmax(xyz,0)
ctr = (mn+mx)/2.0; r = np.max(mx-mn)/2.0
ax.set_xlim(ctr[0]-r, ctr[0]+r)
ax.set_ylim(ctr[1]-r, ctr[1]+r)
ax.set_zlim(ctr[2]-r, ctr[2]+r)
try: ax.set_box_aspect([1,1,1])
except: pass
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
title = ax.set_title(f"3D skeleton — frame 0/{T-1}")

scatters, lines = [], []
t0 = 0
for p in range(P):
    pts = pts_all[t0,p]
    s = ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=25,color="blue",alpha=0.9)
    scatters.append(s)
    person_lines = []
    for (i,j) in edges:
        if i<len(pts) and j<len(pts):
            ln, = ax.plot([pts[i,0],pts[j,0]],
                          [pts[i,1],pts[j,1]],
                          [pts[i,2],pts[j,2]],
                          lw=2,color="red",alpha=0.85)
            person_lines.append(ln)
    lines.append(person_lines)

def update(t):
    title.set_text(f"3D skeleton — frame {t}/{T-1}")
    for p in range(P):
        pts = pts_all[t,p]
        scatters[p]._offsets3d = (pts[:,0],pts[:,1],pts[:,2])
        for ln,(i,j) in zip(lines[p],edges):
            if i<len(pts) and j<len(pts):
                ln.set_data_3d([pts[i,0],pts[j,0]],[pts[i,1],pts[j,1]],[pts[i,2],pts[j,2]])
    return [title]+scatters+[ln for sub in lines for ln in sub]

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