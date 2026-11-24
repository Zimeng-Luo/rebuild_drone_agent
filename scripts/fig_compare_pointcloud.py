"""Render a side-by-side point cloud comparison: COLMAP sparse baseline vs our dense output.

Both clouds are rotated so their dominant ground-plane direction is horizontal,
then normalized to their own bounding box and drawn from the same viewing angle.
The dense cloud is randomly subsampled for rendering only; the point counts in
the panel subtitles are the real totals.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

COLMAP_PLY = "/home/zimeng/rebuild_drone_agent/data/result/baseline_colmap/run/points.ply"
OURS_PLY = ("/home/zimeng/rebuild_drone_agent/data/result/workbench/"
            "a8b1d229d36c4558bcf4c3293fd0b59b/20261114_213843_537471/aligned.ply")
OUT = ("/home/zimeng/rebuild_drone_agent/data/result/report_figures/"
       "fig8_pointcloud_compare.png")

GREY = "#6b7a85"
INK = "#1a2b33"

for candidate in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "/usr/share/fonts/truetype/arphic/uming.ttc"]:
    if os.path.exists(candidate):
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": "white"})


def read_ply(path):
    with open(path, "rb") as handle:
        count = 0
        props = []
        while True:
            line = handle.readline().decode("ascii", "ignore").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        dtype = np.dtype([(p, "<f4") if p in ("x", "y", "z") else (p, "u1") for p in props])
        block = np.fromfile(handle, dtype=dtype, count=count)
    xyz = np.stack([block["x"], block["y"], block["z"]], axis=1).astype(np.float32)
    if "red" in props:
        rgb = np.stack([block["red"], block["green"], block["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        rgb = None
    return xyz, rgb


def upright(xyz):
    """Rotate about the vertical axis so the dominant horizontal spread runs along x."""
    low, high = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    keep = np.all((xyz >= low) & (xyz <= high), axis=1)
    trimmed = xyz[keep]
    flat = trimmed[:, [0, 2]]
    flat = flat - flat.mean(axis=0)
    cov = np.cov(flat.T)
    values, vectors = np.linalg.eigh(cov)
    main = vectors[:, int(np.argmax(values))]
    angle = -np.arctan2(main[1], main[0])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])
    rotated = trimmed @ rot.T
    center = (np.percentile(rotated, 1, axis=0) + np.percentile(rotated, 99, axis=0)) / 2.0
    span = float(np.max(np.percentile(rotated, 99, axis=0) - np.percentile(rotated, 1, axis=0)))
    return (rotated - center) / span, keep


def draw(ax, xyz, rgb, title, sub, size, subsample=None, elev=34, azim=-88):
    if subsample and len(xyz) > subsample:
        pick = np.random.default_rng(5).choice(len(xyz), subsample, replace=False)
        xyz = xyz[pick]
        rgb = rgb[pick] if rgb is not None else None
    ax.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], c=rgb, s=size,
               linewidths=0, marker=".", depthshade=False, rasterized=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.10, 1.0, 0.72))
    ax.set_xlim(-0.62, 0.62)
    ax.set_zlim(-0.26, 0.40)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10.4, color=INK, pad=-2, fontweight="bold")
    ax.text2D(0.5, 0.045, sub, transform=ax.transAxes, ha="center", va="bottom",
              fontsize=8.0, color=GREY)


def main():
    print("loading", flush=True)
    c_xyz, _ = read_ply(COLMAP_PLY)
    o_xyz, o_rgb = read_ply(OURS_PLY)
    print(len(c_xyz), len(o_xyz), flush=True)

    c_n, _ = upright(c_xyz)
    o_n, o_keep = upright(o_xyz)
    o_rgb = o_rgb[o_keep]

    fig = plt.figure(figsize=(6.10, 3.02), dpi=420)

    ax1 = fig.add_axes([0.012, 0.155, 0.462, 0.78], projection="3d")
    draw(ax1, c_n, None, "COLMAP 稀疏重建",
         f"{len(c_xyz):,} 点 · 1595.7 s", size=1.5, subsample=50920)
    ax1.collections[0].set_color(GREY) if hasattr(ax1.collections[0], "set_color") else None
    for coll in ax1.collections:
        coll._uniform_color = (0.42, 0.48, 0.52, 1.0)

    ax2 = fig.add_axes([0.518, 0.155, 0.462, 0.78], projection="3d")
    draw(ax2, o_n, o_rgb, "本文流程稠密重建",
         f"{len(o_xyz):,} 点 · 466.0 s", size=0.10, subsample=620000)

    fig.text(0.5, 0.072,
             "同一段 128.8 s 4K 视频、同一批 81 帧、同一台工作站；两幅各自归一化后按同一视角渲染",
             ha="center", fontsize=7.4, color=GREY)
    fig.text(0.5, 0.022,
             "稠密点云渲染时下采样至 62 万点，副标题中的点数为真实总数",
             ha="center", fontsize=7.4, color=GREY, style="italic")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=420, facecolor="white")
    print("saved", OUT, os.path.getsize(OUT), flush=True)


if __name__ == "__main__":
    main()
