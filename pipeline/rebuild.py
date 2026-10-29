from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from src.logging_utils import configure_logging, configure_task_file_logging
from src.rebuild import (
    load_chunk,
    load_vggt_model,
    rebuild_chunk_to_npz,
    rebuild_npz_to_glb,
    rebuild_npz_to_glb_streaming,
)
from src.task_summary import ResourceStepMonitor, StepSummary


def run_rebuild_chunks(
    chunk_paths: list[str | Path],
    model: torch.nn.Module | None = None,
    model_name_or_path: str = "facebook/VGGT-1B",
    device: str | torch.device | None = None,
    summary_steps: list[StepSummary] | None = None,
) -> list[Path]:
    """Load the VGGT model once and rebuild multiple Chunk files."""
    configure_logging()
    if not chunk_paths:
        return []

    paths = [Path(chunk_path) for chunk_path in chunk_paths]
    configure_task_file_logging(paths[0].parent)

    if model is None:
        with _summary_step(summary_steps, "rebuild_model_load"):
            model = load_vggt_model(
                model_name_or_path=model_name_or_path,
                device=device,
            )

    output_paths: list[Path] = []
    for chunk_path in paths:
        with _summary_step(summary_steps, f"rebuild_chunk_{chunk_path.stem}"):
            chunk = load_chunk(chunk_path)
            output_paths.append(
                rebuild_chunk_to_npz(
                    chunk=chunk,
                    output_path=chunk_path.with_suffix(".npz"),
                    model=model,
                    model_name_or_path=model_name_or_path,
                    device=device,
                )
            )

    return output_paths


def run_rebuild_to_glb(
    npz_path: str | Path,
    output_path: str | Path | None = None,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    prediction_mode: str = "Depthmap and Camera Branch",
    streaming: bool = True,
) -> Path:
    """Convert one reconstruction npz file to GLB beside the npz by default."""
    configure_logging()
    npz_path = Path(npz_path)
    configure_task_file_logging(npz_path.parent)
    converter = rebuild_npz_to_glb_streaming if streaming else rebuild_npz_to_glb
    return converter(
        npz_path=npz_path,
        output_path=output_path,
        conf_thres=conf_thres,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        prediction_mode=prediction_mode,
    )


@contextmanager
def _summary_step(summary_steps: list[StepSummary] | None, name: str):
    if summary_steps is None:
        yield
        return

    with ResourceStepMonitor(name) as monitor:
        yield
    summary_steps.append(monitor.summary())
