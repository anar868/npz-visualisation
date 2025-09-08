#!/usr/bin/env python3
# plot_kp3d_frame.py — visualize ONE 3D skeleton frame on axes (no video)

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

AXIS_ORDER = (0, 2, 1)

FLIPS = (+1, +1, -1)

NPZ   = sys.argv[1] if len(sys.argv) > 1 else "4_1.npz"
FRAME = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # frame index

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

def to_T_P_J_3(kp):
    """Normalize any kp array to (T,P,J,3)."""
    arr = np.asarray(kp)
    coord_axes = [i for i,s in enumerate(arr.shape) if s == 3]
    if coord_axes and coord_axes[-1] != arr.ndim-1:
        arr = np.moveaxis(arr, coord_axes[-1], -1)

    while arr.ndim > 4 and 1 in arr.shape[:-1]:
        arr = np.squeeze(arr)

    return arr 

def set_axes_equal(ax, pts):
    mn = pts.min(0); mx = pts.max(0)
    ctr = (mn+mx)/2.0; r = np.max(mx-mn)/2.0
    ax.set_xlim(ctr[0]-r, ctr[0]+r)
    ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)

def main():
    data = np.load(NPZ, allow_pickle=True)
    kp_TPJ3  = to_T_P_J_3(data["keypoints3d"])   # (T,P,J,3)
    T,P,J,C  = kp_TPJ3.shape
    f = max(0, min(FRAME, T-1))

    pts = kp_TPJ3[f,0,BML87_IDX,:]   # (87,3)
    edges = [tuple(e) for e in BML87_EDGES]
    title = f"BML-MoVi-87 | frame {f}/{T-1}"

    pts = pts[:, AXIS_ORDER] * np.array(FLIPS)[None,:]

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=25, color='b')
    for i,j in edges:
        if i<len(pts) and j<len(pts):
            xi,yi,zi = pts[i]; xj,yj,zj = pts[j]
            ax.plot([xi,xj],[yi,yj],[zi,zj], 'r-', lw=2)
    set_axes_equal(ax, pts)
    ax.set_title(title+f" | persons={P} joints={pts.shape[0]}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.show()

if __name__=="__main__":
    main()