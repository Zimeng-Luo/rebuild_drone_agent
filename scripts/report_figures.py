"""Render figures for the final report from real pipeline output.

Reads the completed multi-chunk run (a8b1d229) and produces:
  fig1_pipeline.png       staged pipeline timing (stacked bar + share)
  fig2_frames.png         frame sampling / keyframe retention / chunk split
  fig3_pointcloud.png     point cloud rendered from aligned.npz world points
  fig4_performance.png    CPU / RAM / GPU peaks by stage
  fig5_space.png          artifact size distribution
All numbers come from task_summary.log and the exported npz.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

RESULT = sys.argv[1] if len(sys.argv) > 1 else "/home/zimeng/rebuild_drone_agent/data/result/workbench/a8b1d229d36c4558bcf4c3293fd0b59b/20261114_213843_537471"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/zimeng/rebuild_drone_agent/data/result/report_figures"

# HIT blue palette, matching the school deck (bottom bar #005375).
GREEN_DARK = "#005375"
GREEN = "#2e75b6"
GREEN_LIGHT = "#9dc3d4"
SAND = "#c9a86a"
GREY = "#6b7a85"
INK = "#1a2b33"

# Chinese glyphs need a CJK face; fall back to an English deck if absent.
for candidate in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "/usr/share/fonts/truetype/arphic/uming.ttc"]:
    if os.path.exists(candidate):
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams.update({
    "axes.edgecolor": "#d5dde3", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GREY, "ytick.color": GREY, "axes.grid": True,
    "grid.color": "#e8eef2", "grid.linewidth": 0.8, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

os.makedirs(OUT, exist_ok=True)
summary = json.load(open(os.path.join(RESULT, "task_summary.log"), encoding="utf-8"))

steps = {s["name"]: s["duration_seconds"] for s in summary["time"]["steps"]}
total = summary["time"]["total"]["duration_seconds"]

# ---------------------------------------------------------------- fig 1
PHASES = [
    ("影像预处理", "读取 · 关键帧筛选 · 天空掩码", "data_preprocess", GREEN_LIGHT),
    ("分块重建", "VGGT 逐块推理", "rebuild_chunks", GREEN),
    ("对齐与拼接", "配准 · 导出 GLB / PLY", "align_and_merge", GREEN_DARK),
]
fig, ax = plt.subplots(figsize=(7.2, 2.5))
left = 0.0
for label, hint, key, color in PHASES:
    value = steps.get(key, 0.0)
    ax.barh([0], [value], left=left, height=0.42, color=color, label=f"{label} {value:.1f}s")
    if value > total * 0.08:
        ax.text(left + value / 2, 0, f"{value / total * 100:.1f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
    left += value
ax.set_yticks([])
ax.set_xlim(0, total)
ax.set_xlabel(f"累计耗时（秒），总计 {total:.1f} s")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False, fontsize=8.5)
ax.grid(axis="y", visible=False)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
fig.savefig(os.path.join(OUT, "fig1_pipeline.png"))
plt.close(fig)

# ---------------------------------------------------------------- fig 2
CHUNKS = summary["parameters"]["run_evidence"]["preprocess"]
per_chunk = CHUNKS["per_chunk_frame_counts"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1.35]})
ax1.bar(["采样帧", "保留关键帧"], [CHUNKS["sampled_frame_count"], CHUNKS["selected_frame_count"]],
        color=[GREEN_LIGHT, GREEN_DARK], width=0.5)
for index, value in enumerate([CHUNKS["sampled_frame_count"], CHUNKS["selected_frame_count"]]):
    ax1.text(index, value, f"{value}", ha="center", va="bottom", fontsize=9, color=INK)
ax1.set_ylim(0, CHUNKS["sampled_frame_count"] * 1.18)
ax1.set_title(f"关键帧保留率 {CHUNKS['selection_retention_ratio'] * 100:.1f}%", fontsize=9)
ax1.grid(axis="x", visible=False)
ax2.bar(range(len(per_chunk)), per_chunk, color=GREEN, width=0.55)
for index, value in enumerate(per_chunk):
    ax2.text(index, value, f"{value}", ha="center", va="bottom", fontsize=8.5)
ax2.set_xticks(range(len(per_chunk)))
ax2.set_xticklabels([f"块 {i + 1}" for i in range(len(per_chunk))])
ax2.set_ylim(0, max(per_chunk) * 1.25)
ax2.set_title(f"分块结果：{len(per_chunk)} 块 / {sum(per_chunk)} 帧", fontsize=9)
ax2.grid(axis="x", visible=False)
for ax in (ax1, ax2):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_frames.png"))
plt.close(fig)

# ---------------------------------------------------------------- fig 3
data = np.load(os.path.join(RESULT, "aligned.npz"))
points = data["world_points_from_depth"]
confidence = data["depth_conf"]
colors = data["images"]
frame_ids = data["frame_ids"]
H, W = points.shape[1], points.shape[2]
# Subsample pixels so plotting stays fast, then drop low-confidence and
# non-finite points exactly the way the exporter does.
stride = 4
flat = points[:, ::stride, ::stride].reshape(-1, 3)
conf = confidence[:, ::stride, ::stride].reshape(-1)
rgb = colors[:, :, ::stride, ::stride].transpose(0, 2, 3, 1).reshape(-1, 3)
valid = np.isfinite(flat).all(axis=1) & (conf > np.percentile(conf[conf > 0], 40) if (conf > 0).any() else conf > 0)
flat, rgb = flat[valid], np.clip(rgb[valid], 0, 1)
print(f"points kept: {len(flat):,} of {points.shape[0] * H * W:,}")

# Two views with a shared, isotropic bounding box so the reconstruction keeps
# its true proportions instead of being stretched by the default 3D aspect.
lo = np.percentile(flat, 1, axis=0)
hi = np.percentile(flat, 99, axis=0)
mid = (lo + hi) / 2
half = np.max(hi - lo) / 2
fig = plt.figure(figsize=(7.2, 3.5))
for index, (elev, azim, title) in enumerate([(26, -62, "俯视三分之四视角"), (2, -90, "水平侧视")]):
    ax = fig.add_subplot(1, 2, index + 1, projection="3d")
    step = 2 if index == 0 else 3
    ax.scatter(flat[::step, 0], flat[::step, 1], flat[::step, 2], c=rgb[::step], s=1.1, linewidths=0, alpha=0.9)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, color=INK, pad=0)
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(OUT, "fig3_pointcloud.png"))
plt.close(fig)
print(f"extent (m): {np.round(hi - lo, 2)}")

# ---------------------------------------------------------------- fig 4
perf = summary["performance"]["steps"]
KEYS = [("data_preprocess", "影像预处理"), ("rebuild_chunks", "分块重建"), ("align_and_merge", "对齐与拼接")]
labels = [label for _, label in KEYS]
gpu = [perf.get(key, {}).get("gpu_memory_allocated_mb", {}).get("peak", 0) for key, _ in KEYS]
rss = [perf.get(key, {}).get("memory_rss_mb", {}).get("peak", 0) for key, _ in KEYS]
cpu = [perf.get(key, {}).get("cpu_percent", {}).get("peak", 0) for key, _ in KEYS]
x = np.arange(len(labels))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.2, 1]})
width = 0.36
ax1.bar(x - width / 2, gpu, width, color=GREEN_DARK, label="显存峰值")
ax1.bar(x + width / 2, rss, width, color=SAND, label="内存峰值")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8.5)
ax1.set_ylabel("MB")
ax1.legend(frameon=False, fontsize=8)
ax1.set_title("显存与内存峰值", fontsize=9)
ax2.bar(x, cpu, 0.5, color=GREEN)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8.5)
ax2.set_ylabel("CPU %")
ax2.set_title("CPU 峰值（多核合计）", fontsize=9)
for ax in (ax1, ax2):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_performance.png"))
plt.close(fig)

# ---------------------------------------------------------------- fig 5
space = summary["space"]["by_suffix"]
items = sorted(((suffix, value["mb"]) for suffix, value in space.items() if value.get("mb", 0) >= 0.1), key=lambda pair: -pair[1])
fig, ax = plt.subplots(figsize=(4.6, 1.9))
names = [name for name, _ in items]
values = [value for _, value in items]
bars = ax.barh(names[::-1], values[::-1], color=[GREEN_LIGHT, SAND, GREEN, GREEN_DARK][:len(names)][::-1], height=0.55)
for bar, value in zip(bars, values[::-1]):
    ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.1f} MB", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, max(values) * 1.28)
ax.set_xlabel("MB")
ax.grid(axis="y", visible=False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.savefig(os.path.join(OUT, "fig5_space.png"))
plt.close(fig)

print("figures written to", OUT)
for name in sorted(os.listdir(OUT)):
    print(" ", name, os.path.getsize(os.path.join(OUT, name)), "bytes")
