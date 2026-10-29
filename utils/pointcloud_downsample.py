from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
import struct
from typing import Any, BinaryIO

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PointCloudDownsampleConfig:
    enabled: bool = True
    voxel_size: float = 0.02
    min_keep_ratio: float = 0.5


def downsample_point_cloud_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    enabled: bool = True,
    voxel_size: float = 0.02,
    min_keep_ratio: float = 0.5,
) -> Path:
    """Downsample one GLB or PLY point cloud file and save a new file."""
    input_path = Path(input_path)
    output_path = _default_output_path(input_path) if output_path is None else Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not enabled:
        shutil.copy2(input_path, output_path)
        logger.info("点云下采样关闭，已复制原文件: input_path=%s, output_path=%s", input_path, output_path)
        return output_path

    logger.info(
        "点云下采样开始: input_path=%s, output_path=%s, voxel_size=%.6g, min_keep_ratio=%.3f",
        input_path,
        output_path,
        voxel_size,
        min_keep_ratio,
    )
    logger.info("点云加载开始: input_path=%s", input_path)
    points, colors = load_point_cloud(input_path)
    original_count = points.shape[0]
    logger.info(
        "点云加载完成: input_path=%s, points=%d, has_color=%s",
        input_path,
        original_count,
        colors is not None,
    )
    if original_count == 0:
        raise ValueError(f"{input_path} contains no point vertices")

    logger.info("点云有限值过滤开始: points=%d", original_count)
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    colors = None if colors is None else colors[finite_mask]
    logger.info(
        "点云有限值过滤完成: before_points=%d, after_points=%d, removed_points=%d",
        original_count,
        points.shape[0],
        original_count - points.shape[0],
    )

    points, colors, used_voxel_size = voxel_grid_downsample_with_keep_limit(
        points,
        colors,
        voxel_size=voxel_size,
        min_keep_ratio=min_keep_ratio,
    )

    logger.info("点云保存开始: output_path=%s, points=%d", output_path, points.shape[0])
    save_point_cloud(output_path, points, colors)
    logger.info("点云保存完成: output_path=%s", output_path)
    logger.info(
        "点云下采样完成: input_path=%s, output_path=%s, original_points=%d, "
        "output_points=%d, keep_ratio=%.3f, requested_voxel_size=%.6g, used_voxel_size=%.6g, min_keep_ratio=%.3f",
        input_path,
        output_path,
        original_count,
        points.shape[0],
        points.shape[0] / original_count,
        voxel_size,
        used_voxel_size,
        min_keep_ratio,
    )
    return output_path


def load_point_cloud(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.uint8] | None]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return _load_ply(path)
    if suffix in (".glb", ".gltf"):
        return _load_glb(path)
    raise ValueError(f"unsupported point cloud file type: {path.suffix}")


def save_point_cloud(
    path: str | Path,
    points: NDArray[np.floating],
    colors: NDArray[np.uint8] | None,
) -> Path:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ply":
        _save_ply(path, points, colors)
    elif suffix == ".glb":
        _save_glb(path, points, colors)
    else:
        raise ValueError(f"unsupported point cloud file type: {path.suffix}")
    return path


def voxel_grid_downsample_with_keep_limit(
    points: NDArray[np.floating],
    colors: NDArray[np.uint8] | None = None,
    *,
    voxel_size: float,
    min_keep_ratio: float,
) -> tuple[NDArray[np.float64], NDArray[np.uint8] | None, float]:
    """Voxel-downsample while keeping at least min_keep_ratio of points."""
    points = np.asarray(points, dtype=np.float64)
    _validate_min_keep_ratio(min_keep_ratio)
    if voxel_size <= 0.0 or points.shape[0] == 0:
        sampled_points, sampled_colors = voxel_grid_downsample(points, colors, voxel_size=voxel_size)
        return sampled_points, sampled_colors, voxel_size

    min_keep_points = int(np.ceil(points.shape[0] * min_keep_ratio))
    logger.info(
        "voxel grid保留比例限制开始: points=%d, requested_voxel_size=%.6g, min_keep_ratio=%.3f, min_keep_points=%d",
        points.shape[0],
        voxel_size,
        min_keep_ratio,
        min_keep_points,
    )

    best_points, best_colors = voxel_grid_downsample(points, colors, voxel_size=voxel_size)
    best_voxel_size = voxel_size
    if best_points.shape[0] >= min_keep_points:
        logger.info(
            "voxel grid保留比例已满足: output_points=%d, keep_ratio=%.3f, used_voxel_size=%.6g",
            best_points.shape[0],
            best_points.shape[0] / points.shape[0],
            best_voxel_size,
        )
        return best_points, best_colors, best_voxel_size

    low = 0.0
    high = float(voxel_size)
    for iteration in range(12):
        candidate_voxel_size = (low + high) / 2.0
        candidate_points, candidate_colors = voxel_grid_downsample(
            points,
            colors,
            voxel_size=candidate_voxel_size,
        )
        keep_ratio = candidate_points.shape[0] / points.shape[0]
        logger.info(
            "voxel grid保留比例搜索: iteration=%d, voxel_size=%.6g, output_points=%d, keep_ratio=%.3f",
            iteration + 1,
            candidate_voxel_size,
            candidate_points.shape[0],
            keep_ratio,
        )
        if candidate_points.shape[0] >= min_keep_points:
            best_points = candidate_points
            best_colors = candidate_colors
            best_voxel_size = candidate_voxel_size
            low = candidate_voxel_size
        else:
            high = candidate_voxel_size

    logger.info(
        "voxel grid保留比例限制完成: output_points=%d, keep_ratio=%.3f, used_voxel_size=%.6g",
        best_points.shape[0],
        best_points.shape[0] / points.shape[0],
        best_voxel_size,
    )
    return best_points, best_colors, best_voxel_size


def voxel_grid_downsample(
    points: NDArray[np.floating],
    colors: NDArray[np.uint8] | None = None,
    *,
    voxel_size: float,
) -> tuple[NDArray[np.float64], NDArray[np.uint8] | None]:
    """Downsample points by averaging each occupied voxel."""
    points = np.asarray(points, dtype=np.float64)
    if voxel_size <= 0.0 or points.shape[0] == 0:
        logger.info(
            "voxel grid跳过: voxel_size=%.6g, points=%d",
            voxel_size,
            points.shape[0],
        )
        return points, colors

    logger.info("voxel grid开始: points=%d, voxel_size=%.6g", points.shape[0], voxel_size)
    logger.info("voxel grid计算体素索引开始")
    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    logger.info("voxel grid合并体素开始")
    _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    voxel_count = int(inverse.max()) + 1
    logger.info("voxel grid聚合开始: voxel_count=%d", voxel_count)

    counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)
    sampled_points = np.empty((voxel_count, 3), dtype=np.float64)
    for axis in range(3):
        sampled_points[:, axis] = np.bincount(
            inverse,
            weights=points[:, axis],
            minlength=voxel_count,
        ) / counts

    if colors is None:
        logger.info(
            "voxel grid完成: before_points=%d, after_points=%d",
            points.shape[0],
            sampled_points.shape[0],
        )
        return sampled_points, None

    logger.info("voxel grid颜色聚合开始")
    color_values = np.asarray(colors, dtype=np.float64)
    sampled_colors = np.empty((voxel_count, 3), dtype=np.uint8)
    for channel in range(3):
        sampled_colors[:, channel] = np.rint(
            np.bincount(inverse, weights=color_values[:, channel], minlength=voxel_count) / counts
        ).clip(0, 255).astype(np.uint8)
    logger.info(
        "voxel grid完成: before_points=%d, after_points=%d",
        points.shape[0],
        sampled_points.shape[0],
    )
    return sampled_points, sampled_colors


def _load_ply(path: Path) -> tuple[NDArray[np.float64], NDArray[np.uint8] | None]:
    with path.open("rb") as file:
        header_lines = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"{path} has no PLY end_header")
            header_lines.append(line.decode("ascii").strip())
            if header_lines[-1] == "end_header":
                break

        if not header_lines or header_lines[0] != "ply":
            raise ValueError(f"{path} is not a PLY file")

        fmt = _ply_header_value(header_lines, "format")
        vertex_count = _ply_vertex_count(header_lines)
        vertex_properties = _ply_vertex_properties(header_lines)

        if fmt == "binary_little_endian 1.0":
            dtype = np.dtype([(name, _PLY_DTYPE[type_name]) for type_name, name in vertex_properties])
            records = np.fromfile(file, dtype=dtype, count=vertex_count)
        elif fmt == "ascii 1.0":
            records = _read_ascii_ply_vertices(file, vertex_count, vertex_properties)
        else:
            raise ValueError(f"unsupported PLY format: {fmt}")

    points = np.column_stack([records["x"], records["y"], records["z"]]).astype(np.float64)
    if all(name in records.dtype.names for name in ("red", "green", "blue")):
        colors = np.column_stack([records["red"], records["green"], records["blue"]]).astype(np.uint8)
    else:
        colors = None
    return points, colors


def _save_ply(path: Path, points: NDArray[np.floating], colors: NDArray[np.uint8] | None) -> None:
    colors = _default_colors(points, colors)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    records = np.empty(points.shape[0], dtype=dtype)
    records["x"] = points[:, 0].astype("<f4", copy=False)
    records["y"] = points[:, 1].astype("<f4", copy=False)
    records["z"] = points[:, 2].astype("<f4", copy=False)
    records["red"] = colors[:, 0]
    records["green"] = colors[:, 1]
    records["blue"] = colors[:, 2]
    with path.open("wb") as file:
        file.write(header)
        records.tofile(file)


def _load_glb(path: Path) -> tuple[NDArray[np.float64], NDArray[np.uint8] | None]:
    with path.open("rb") as file:
        magic, version, _ = struct.unpack("<4sII", _read_exact(file, 12))
        if magic != b"glTF" or version != 2:
            raise ValueError(f"{path} is not a GLB v2 file")

        json_payload: dict[str, Any] | None = None
        bin_payload = b""
        while True:
            chunk_header = file.read(8)
            if not chunk_header:
                break
            chunk_length, chunk_type = struct.unpack("<I4s", chunk_header)
            chunk_data = _read_exact(file, chunk_length)
            if chunk_type == b"JSON":
                json_payload = json.loads(chunk_data.decode("utf-8"))
            elif chunk_type == b"BIN\x00":
                bin_payload = chunk_data

    if json_payload is None:
        raise ValueError(f"{path} has no JSON chunk")

    primitive = json_payload["meshes"][0]["primitives"][0]
    position_accessor = json_payload["accessors"][primitive["attributes"]["POSITION"]]
    points = _read_glb_accessor(json_payload, bin_payload, position_accessor).astype(np.float64)

    colors = None
    color_accessor_index = primitive["attributes"].get("COLOR_0")
    if color_accessor_index is not None:
        color_values = _read_glb_accessor(json_payload, bin_payload, json_payload["accessors"][color_accessor_index])
        colors = _glb_colors_to_uint8(color_values, json_payload["accessors"][color_accessor_index])
    return points, colors


def _save_glb(path: Path, points: NDArray[np.floating], colors: NDArray[np.uint8] | None) -> None:
    colors = _default_colors(points, colors)
    vertex_dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("pad", "u1")]
    )
    records = np.zeros(points.shape[0], dtype=vertex_dtype)
    records["x"] = points[:, 0].astype("<f4", copy=False)
    records["y"] = points[:, 1].astype("<f4", copy=False)
    records["z"] = points[:, 2].astype("<f4", copy=False)
    records["red"] = colors[:, 0]
    records["green"] = colors[:, 1]
    records["blue"] = colors[:, 2]
    bin_payload = records.tobytes()
    json_payload = _build_glb_json(points, len(bin_payload))
    json_padding = _align4(len(json_payload)) - len(json_payload)
    bin_padding = _align4(len(bin_payload)) - len(bin_payload)
    total_length = 12 + 8 + len(json_payload) + json_padding + 8 + len(bin_payload) + bin_padding

    with path.open("wb") as file:
        file.write(struct.pack("<4sII", b"glTF", 2, total_length))
        file.write(struct.pack("<I4s", len(json_payload) + json_padding, b"JSON"))
        file.write(json_payload)
        file.write(b" " * json_padding)
        file.write(struct.pack("<I4s", len(bin_payload) + bin_padding, b"BIN\x00"))
        file.write(bin_payload)
        file.write(b"\x00" * bin_padding)


def _read_glb_accessor(
    gltf: dict[str, Any],
    bin_payload: bytes,
    accessor: dict[str, Any],
) -> NDArray[Any]:
    component_count = _GLB_ACCESSOR_COMPONENT_COUNTS.get(accessor["type"])
    if component_count is None:
        raise ValueError(f"unsupported GLB accessor type: {accessor['type']}")
    component_dtype = _GLB_COMPONENT_DTYPE[accessor["componentType"]]
    component_size = np.dtype(component_dtype).itemsize
    count = int(accessor["count"])
    buffer_view = gltf["bufferViews"][accessor["bufferView"]]
    offset = int(buffer_view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(buffer_view.get("byteStride", component_size * component_count))
    data = np.frombuffer(bin_payload, dtype=np.uint8)
    if stride == component_size * component_count:
        return (
            np.frombuffer(bin_payload, dtype=component_dtype, count=count * component_count, offset=offset)
            .reshape(count, component_count)
            .copy()
        )

    values = np.ndarray(
        shape=(count, component_count),
        dtype=component_dtype,
        buffer=data,
        offset=offset,
        strides=(stride, component_size),
    )
    return values.copy()


def _build_glb_json(points: NDArray[np.floating], bin_length: int) -> bytes:
    bounds_min = np.min(points, axis=0).astype(float).tolist()
    bounds_max = np.max(points, axis=0).astype(float).tolist()
    payload = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": 0}]}],
        "buffers": [{"byteLength": bin_length}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": bin_length, "byteStride": 16, "target": 34962}
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(points.shape[0]),
                "type": "VEC3",
                "min": bounds_min,
                "max": bounds_max,
            },
            {
                "bufferView": 0,
                "byteOffset": 12,
                "componentType": 5121,
                "count": int(points.shape[0]),
                "type": "VEC3",
                "normalized": True,
            },
        ],
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _glb_colors_to_uint8(values: NDArray[Any], accessor: dict[str, Any]) -> NDArray[np.uint8]:
    values = np.asarray(values)
    if values.shape[1] < 3:
        raise ValueError(f"GLB color accessor must have at least 3 components, got {values.shape[1]}")
    values = values[:, :3]
    if accessor["componentType"] == 5121:
        return np.asarray(values, dtype=np.uint8)
    if accessor.get("normalized"):
        return np.rint(np.asarray(values, dtype=np.float64) * 255.0).clip(0, 255).astype(np.uint8)
    return np.asarray(values, dtype=np.float64).clip(0, 255).astype(np.uint8)


def _default_colors(points: NDArray[np.floating], colors: NDArray[np.uint8] | None) -> NDArray[np.uint8]:
    if colors is not None:
        return np.asarray(colors, dtype=np.uint8)
    return np.full((points.shape[0], 3), 255, dtype=np.uint8)


def _default_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_downsampled{path.suffix}")


def _validate_min_keep_ratio(min_keep_ratio: float) -> None:
    if not 0.0 < min_keep_ratio <= 1.0:
        raise ValueError(f"min_keep_ratio must be in (0, 1], got {min_keep_ratio}")


def _log_progress(stage: str, current: int, total: int) -> None:
    if total <= 0:
        return
    step = max(total // 10, 1_000_000, 1)
    if current < total and current % step != 0:
        return
    percent = current / total * 100.0
    logger.info("%s进度: %d/%d (%.1f%%)", stage, current, total, percent)


def _read_exact(file: BinaryIO, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ValueError("unexpected end of file")
    return data


def _ply_header_value(header_lines: list[str], key: str) -> str:
    prefix = f"{key} "
    for line in header_lines:
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise ValueError(f"PLY header missing {key}")


def _ply_vertex_count(header_lines: list[str]) -> int:
    for line in header_lines:
        if line.startswith("element vertex "):
            return int(line.split()[2])
    raise ValueError("PLY header missing vertex element")


def _ply_vertex_properties(header_lines: list[str]) -> list[tuple[str, str]]:
    properties = []
    in_vertex = False
    for line in header_lines:
        if line.startswith("element "):
            in_vertex = line.startswith("element vertex ")
            continue
        if in_vertex and line.startswith("property "):
            _, type_name, name = line.split()[:3]
            if type_name not in _PLY_DTYPE:
                raise ValueError(f"unsupported PLY property type: {type_name}")
            properties.append((type_name, name))
    required = {"x", "y", "z"}
    names = {name for _, name in properties}
    if not required.issubset(names):
        raise ValueError(f"PLY vertex properties must include {sorted(required)}")
    return properties


def _read_ascii_ply_vertices(
    file: BinaryIO,
    vertex_count: int,
    vertex_properties: list[tuple[str, str]],
) -> NDArray[Any]:
    text = file.read().decode("ascii").splitlines()[:vertex_count]
    rows = [line.split() for line in text if line.strip()]
    dtype = np.dtype([(name, _PLY_DTYPE[type_name]) for type_name, name in vertex_properties])
    records = np.empty(vertex_count, dtype=dtype)
    for row_index, row in enumerate(rows):
        for col_index, (_, name) in enumerate(vertex_properties):
            records[name][row_index] = row[col_index]
    return records


def _align4(value: int) -> int:
    return (value + 3) & ~3


_PLY_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}

_GLB_COMPONENT_DTYPE = {
    5120: "i1",
    5121: "u1",
    5122: "<i2",
    5123: "<u2",
    5125: "<u4",
    5126: "<f4",
}

_GLB_ACCESSOR_COMPONENT_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}
