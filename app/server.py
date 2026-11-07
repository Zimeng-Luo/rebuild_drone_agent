"""Local trusted workbench API. Start with python -m app.server."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Literal
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError
from starlette.datastructures import UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from app import __version__
from app.jobs import BusyError, JobManager, tail

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpeg", ".mpg", ".mts", ".m2ts"}
MAX_UPLOAD_BYTES = 8 * 1024**3
MAX_IMAGE_BYTES = 100 * 1024**2
MAX_UPLOAD_FILES = 10000
CHUNK_BYTES = 1024**2


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_path: str = Field(min_length=1, max_length=4096)
    read_type: Literal["video", "image"]
    name: str = Field(default="", max_length=200)
    image_sample_interval: int = Field(default=3, ge=1, le=1000, strict=True)
    remove_mask_enable: StrictBool = False
    confidence_percentile: float = Field(default=0, ge=0, le=100, allow_inf_nan=False)
    downsample_enable: StrictBool = False


def validate_input(root: Path, raw: str, read_type: str) -> Path:
    try:
        path = Path(raw).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if read_type == "video":
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError("video input must be an existing supported video file")
            if path.stat().st_size == 0:
                raise ValueError("video input is empty")
        else:
            if not path.is_dir():
                raise ValueError("image input must be an existing image directory")
            count = 0
            with os.scandir(path) as entries:
                for entry in entries:
                    suffix = Path(entry.name).suffix.lower()
                    if not entry.is_file():
                        continue
                    if suffix in VIDEO_EXTENSIONS:
                        raise ValueError("image directory cannot contain mixed video and image inputs")
                    if suffix in IMAGE_EXTENSIONS:
                        if entry.stat().st_size == 0:
                            raise ValueError(f"image input contains an empty file: {entry.name}")
                        count += 1
            if count < 2:
                raise ValueError("image input must contain at least 2 images directly in the directory")
        return path
    except (OSError, ValueError) as exc:
        raise HTTPException(422, detail=f"Invalid input: {exc}") from exc


def discover_inputs(root: Path) -> list[dict]:
    """Bounded metadata-only discovery, stopping at the first image directory."""
    base = root / "data/example"
    if not base.is_dir():
        return []
    items = []
    pending = [(base, 0)]
    visited = 0
    while pending and visited < 128:
        folder, depth = pending.pop(0)
        visited += 1
        images = 0
        child_dirs = []
        videos = []
        try:
            with os.scandir(folder) as entries:
                for index, entry in enumerate(entries):
                    if index >= 2000:
                        break
                    if entry.is_symlink():
                        continue
                    if entry.is_dir():
                        if depth < 5:
                            child_dirs.append((Path(entry.path), depth + 1))
                    elif Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS:
                        images += 1
                        if images >= 2 and folder != base and not videos:
                            break
                    elif Path(entry.name).suffix.lower() in VIDEO_EXTENSIONS:
                        videos.append(dict(path=str(Path(entry.path).resolve()), name=entry.name, type="video"))
            if images >= 2 and not videos:
                items.append(dict(path=str(folder.resolve()), name=folder.name, type="image"))
            else:
                items.extend(videos)
                pending.extend(child_dirs[:128 - visited])
        except OSError:
            continue
    return sorted(items, key=lambda item: (item["type"], item["name"]))


def gpu_info() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        first = result.stdout.strip().splitlines()[0]
        name, memory = first.rsplit(",", 1)
        return dict(available=True, name=name.strip(), memory_total_gb=round(float(memory) / 1024, 2))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return dict(available=False, name=None)


def create_app(root: Path | str = ROOT, worker_command_factory: Callable[[Path], list[str]] | None = None,
               *, max_concurrent: int = 1, max_queue: int = 8, gpu_devices: str | list[str] | None = None) -> FastAPI:
    root = Path(root).resolve()
    manager = JobManager(root, worker_command_factory, max_concurrent=max_concurrent,
                         max_queue=max_queue, gpu_devices=gpu_devices)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        manager.start()
        try:
            yield
        finally:
            await run_in_threadpool(manager.close)

    application = FastAPI(title="Drone Reconstruction Workbench", version=__version__, lifespan=lifespan)
    application.state.manager = manager

    @application.middleware("http")
    async def local_browser_boundary(request: Request, call_next):
        # No cross-origin browser writes to this trusted local service.
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            from urllib.parse import urlsplit
            if urlsplit(origin).netloc != request.headers.get("host"):
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "Cross-origin writes are disabled for this local workbench"}, status_code=403)
        return await call_next(request)

    @application.get("/api/system")
    def system():
        state = manager.scheduler_state()
        return dict(version=__version__, gpu=gpu_info(), disk=dict(free_gb=round(shutil.disk_usage(root).free / 1024**3, 2)),
                    **state)

    @application.get("/api/inputs")
    def inputs():
        return dict(items=discover_inputs(root))

    @application.post("/api/uploads")
    async def upload(request: Request):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 16 * 1024**2:
            raise HTTPException(413, detail="Upload exceeds the 8 GiB total limit")
        # Bound the multipart parser itself, including requests without Content-Length.
        original_receive = request._receive
        received = 0
        async def bounded_receive():
            nonlocal received
            message = await original_receive()
            received += len(message.get("body", b""))
            if received > MAX_UPLOAD_BYTES + 16 * 1024**2:
                # MultiPartException makes Starlette close every temporary spool file.
                raise MultiPartException("Upload exceeds the 8 GiB total limit")
            return message
        request._receive = bounded_receive
        folder = root / "data/uploads" / uuid.uuid4().hex
        try:
            async with request.form(max_files=MAX_UPLOAD_FILES, max_fields=0) as form:
                entries = form.getlist("files")
                if not entries or any(key != "files" for key in form.keys()) or not all(isinstance(item, UploadFile) for item in entries):
                    raise HTTPException(422, detail="Provide files using the repeated multipart field 'files'")
                names = []
                kinds = set()
                for file in entries:
                    name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
                    if name in {"", ".", ".."} or any(ord(char) < 32 for char in name) or len(name.encode("utf-8")) > 240:
                        raise HTTPException(422, detail="Each file needs a valid basename shorter than 240 UTF-8 bytes")
                    suffix = Path(name).suffix.lower()
                    if suffix in VIDEO_EXTENSIONS:
                        kinds.add("video")
                    elif suffix in IMAGE_EXTENSIONS:
                        kinds.add("image")
                    else:
                        raise HTTPException(422, detail=f"Unsupported file extension: {name}")
                    if name.casefold() in {existing.casefold() for existing in names}:
                        raise HTTPException(422, detail=f"Duplicate upload basename: {name}")
                    names.append(name)
                if len(kinds) != 1:
                    raise HTTPException(422, detail="Do not mix videos and images in one upload")
                read_type = kinds.pop()
                if read_type == "video" and len(entries) != 1:
                    raise HTTPException(422, detail="Upload exactly 1 video")
                if read_type == "image" and len(entries) < 2:
                    raise HTTPException(422, detail="Upload at least 2 images")
                folder.mkdir(parents=True)
                total = 0
                for file, name in zip(entries, names):
                    file_size = 0
                    with (folder / name).open("xb") as target:
                        while chunk := await file.read(CHUNK_BYTES):
                            total += len(chunk)
                            file_size += len(chunk)
                            if total > MAX_UPLOAD_BYTES:
                                raise HTTPException(413, detail="Upload exceeds the 8 GiB total limit")
                            if read_type == "image" and file_size > MAX_IMAGE_BYTES:
                                raise HTTPException(413, detail=f"Image exceeds the 100 MiB per-file limit: {name}")
                            await run_in_threadpool(target.write, chunk)
                    if file_size == 0:
                        raise HTTPException(422, detail=f"Empty file is not allowed: {name}")
                input_path = folder / names[0] if read_type == "video" else folder
                return dict(input_path=str(input_path), read_type=read_type,
                            label=names[0] if read_type == "video" else f"{len(names)} images")
        except Exception as exc:
            if folder.exists():
                await run_in_threadpool(shutil.rmtree, folder)
            if isinstance(exc, StarletteHTTPException) and exc.detail == "Upload exceeds the 8 GiB total limit":
                raise HTTPException(413, detail=exc.detail) from exc
            raise

    @application.post("/api/jobs", status_code=201)
    def submit(body: JobRequest):
        path = validate_input(root, body.input_path, body.read_type)
        options = body.model_dump(exclude={"input_path", "read_type", "name"})
        try:
            return manager.submit(str(path), body.read_type, body.name.strip(), options)
        except BusyError as exc:
            raise HTTPException(409, detail=str(exc)) from exc

    @application.get("/api/jobs")
    def jobs():
        return dict(jobs=manager.list())

    def get_job(job_id: str):
        try:
            return manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, detail="Job not found") from exc

    @application.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return get_job(job_id)

    @application.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        get_job(job_id)
        return manager.cancel(job_id)

    @application.post("/api/jobs/{job_id}/retry", status_code=201)
    def retry(job_id: str):
        original = get_job(job_id)
        if original["status"] not in {"failed", "cancelled", "interrupted"}:
            raise HTTPException(409, detail="Only failed, cancelled or interrupted jobs can be retried")
        try:
            body = JobRequest.model_validate(dict(**original["options"], input_path=original["input_path"],
                                                  read_type=original["read_type"], name=original["name"]))
        except (ValidationError, KeyError, TypeError) as exc:
            raise HTTPException(422, detail=f"Original job request is invalid: {exc}") from exc
        path = validate_input(root, body.input_path, body.read_type)
        try:
            return manager.submit(str(path), body.read_type, body.name.strip(),
                                  body.model_dump(exclude={"input_path", "read_type", "name"}), retry_of=job_id)
        except BusyError as exc:
            raise HTTPException(409, detail=str(exc)) from exc

    @application.get("/api/jobs/{job_id}/logs")
    def logs(job_id: str, limit: int = Query(default=200, ge=1, le=1000)):
        get_job(job_id)
        return dict(lines=tail(manager.jobs_dir / job_id / "stdout.log", limit))

    @application.get("/api/jobs/{job_id}/artifacts/{relative:path}")
    def artifact(job_id: str, relative: str):
        get_job(job_id)
        try:
            path = manager.artifact(job_id, relative)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise HTTPException(404, detail="Artifact not found or not allowed") from exc
        return FileResponse(path, filename=path.name)

    @application.get("/{relative:path}", include_in_schema=False)
    def static(relative: str):
        if relative == "api" or relative.startswith("api/"):
            raise HTTPException(404, detail="API route not found")
        static_dir = (root / "app/static").resolve()
        requested = static_dir / (relative or "index.html")
        try:
            resolved = requested.resolve()
            if not resolved.is_relative_to(static_dir) or ".." in relative.split("/"):
                raise HTTPException(404, detail="Static file not found")
            if resolved.is_file():
                return FileResponse(resolved)
        except (OSError, ValueError):
            raise HTTPException(404, detail="Static file not found")
        if relative:
            raise HTTPException(404, detail="Static file not found")
        return HTMLResponse("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Build frontend</title>"
                            "<body><h1>工作台前端尚未构建</h1><p>请在 frontend 目录安装前端依赖并运行 npm run build，"
                            "将构建产物写入 app/static，然后刷新此页。</p><p>Backend API is ready at /docs.</p></body></html>", status_code=503)

    return application


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trusted local reconstruction workbench; run a single server process")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-concurrent", type=int, default=1, help="Concurrent jobs (one GPU per job; explicit pool required above 1)")
    parser.add_argument("--max-queue", type=int, default=8, help="Maximum waiting jobs, excluding running jobs")
    parser.add_argument("--gpu-devices", help="Comma-separated physical GPU indices or full GPU UUIDs, never mixed; inspect nvidia-smi first")
    args = parser.parse_args()
    try:
        application = create_app(max_concurrent=args.max_concurrent, max_queue=args.max_queue,
                                 gpu_devices=args.gpu_devices)
    except ValueError as exc:
        parser.error(str(exc))
    import uvicorn
    uvicorn.run(application, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
