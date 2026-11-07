# 航境 · AEROSCAPE

面向无人机影像的三维重建工作台。输入一个航拍视频或连续照片，完成关键帧筛选、重叠分块、VGGT 重建、Sim3 对齐与点云导出，并在浏览器中查看任务、日志和三维结果。

这是大一年度项目的结题工程整理。沿用 VGGT-1B、FastAPI 与 Three.js 技术栈；当前重点为任务调度、操作体验、稳定性和使用文档。

## 已有能力

- 视频输入与图像目录输入；视频默认按 3 帧/秒提取，图像可设置采样间隔。
- 两遍关键帧筛选、30 帧分块及 15 帧重叠；可选 SegFormer 天空过滤。
- VGGT 深度与相机预测，相邻块 Sim3 对齐，以及非相邻块回环候选与图优化。
- GLB / PLY 点云导出、置信度百分位过滤、可选体素降采样。
- 中文工作台：本地上传、服务器示例、路径输入、有界任务队列、可配置并发、阶段与日志、取消、失败重试、历史记录、结果下载。
- Three.js 查看器：按需加载 GLB / PLY、旋转平移缩放、重置视角、网格与背景切换、点大小及全屏。

## 快速启动

当前服务目标环境是 Linux，Python 3.10+。任务管理使用 Linux 进程组、`flock` 和父进程退出信号。GPU 重建建议使用有足够显存的 NVIDIA 显卡；GPU 名称和剩余磁盘可在页面查看。首次推理可能下载 VGGT 权重；天空过滤启用时还需下载 SegFormer。

在现有服务器上：

```bash
cd /home/zimeng/rebuild_drone_agent
conda activate rebuild_drone_agent
python -m pip install -r requirements-dev.txt
cd frontend
npm ci
npm run check
npm run build
cd ..
nvidia-smi
# 根据实时占用选择空闲卡；下面以物理 0 号卡为例。
CUDA_VISIBLE_DEVICES=0 python -m app.server --host 127.0.0.1 --port 7860
```

浏览器打开 <http://127.0.0.1:7860>。若服务位于远端，在本机另开终端建立 SSH 转发：

```bash
ssh -p 21102 -L 7860:127.0.0.1:7860 zimeng@9aaf14c8c684a539.natapp.cc
```

新环境请先根据 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 安装匹配 CUDA 的 `torch` / `torchvision`，再安装项目依赖。前端构建建议 Node.js 22.12+ 或 24 LTS（Vite 7 要求 Node 20.19+ / 22.12+）。构建产物自动写入后端静态目录；只修改前端源代码后也需要重新构建。

默认 Hugging Face 端点为 `https://hf-mirror.com`，可在启动前通过 `HF_ENDPOINT` 覆盖。已有缓存权重会复用。

## 使用方式

1. 选择本地上传、服务器示例，或输入服务器上的视频文件 / 图像目录路径。
2. 输入可选任务名称，设置图像采样间隔。视频采样率由配置中的 `VIDEO_EXTRACT_FPS` 控制，图像采样间隔不会改变视频采样率。
3. 按需开启天空过滤、点云降采样，或调整置信度过滤百分位。百分位为 0 表示不额外过滤；数值增加会减少保留点。
4. 启动后查看真实流水线阶段、设备与日志；默认 1 个执行槽位、最多 8 个等待任务，等待任务按提交顺序推进。可取消等待或运行任务。
5. 完成后点击产物旁的“查看”加载模型，或直接下载。切换历史任务不会自动下载模型；失败、取消或中断的任务可点击“重试”，保留原记录并创建新任务。

上传要求：单次一个视频，或至少两张图片；不能混合视频与图片。最多 10000 个文件，总体积不超过 8 GiB，单张图片不超过 100 MiB。上传只检查格式和大小，文件损坏等问题由实际解码阶段报告。

服务器示例从 `data/example` 中扫描。路径输入指向运行服务的机器，可使用其上已有素材。此版用于本机或 SSH 转发后的受信访问；使用一个后端进程管理本项目的队列，无登录系统。多卡并发不协调其他用户或其他项目对 GPU 的使用，运行前仍需检查占用。

三维导出主要是点云，并不等同于带纹理网格。页面的参考网格不代表真实地理尺度。单目重建的尺度与坐标需要额外标定；数据结构预留了 GPS 字段，但当前图片与视频加载器未读取实际 GPS，不能声称已实现地理定位。

## 队列与多卡并发

默认 `--max-concurrent 1 --max-queue 8`。执行槽位已满时，新任务进入等待队列；等待队列也满时明确拒绝新任务。`--max-queue 0` 表示只接受能立即执行的任务。取消运行任务时，先终止并回收该任务进程组，再让下一个任务使用槽位；取消排队任务不影响运行中的任务。

要同时执行两个任务，先确认两张卡都空闲，再显式配置独立 GPU 池，例如：

```bash
nvidia-smi
# 仅当物理 0、1 号卡当时均空闲时使用。
python -m app.server --host 127.0.0.1 --port 7860 --gpu-devices 0,1 --max-concurrent 2 --max-queue 8
```

每个执行槽位只绑定一张卡，worker 通过 `CUDA_VISIBLE_DEVICES` 只看到分配给自己的 GPU。也可使用完整 GPU UUID，但一个池中不能混用编号与 UUID，不能重复设备；并发数不能超过池的设备数。没有 `--gpu-devices` 时，只接受单卡 `CUDA_VISIBLE_DEVICES` 配置。未明确指定可用设备时，页面可正常浏览历史和下载，提交任务会提示先配置 GPU。

失败、取消或中断后点击“重试”，会重新检查输入是否存在、参数是否合法，并生成新任务 ID；原任务及其日志保留。等待与运行任务在服务关闭时均标记为 interrupted，重启后不会自动恢复，避免在未检查 GPU 占用的情况下开始推理。队列与历史属于一个后端进程，不要使用多个 Uvicorn worker。

## 命令行

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python main.py data/example/01.mp4 --read-type video
CUDA_VISIBLE_DEVICES=0 python main.py /absolute/path/to/images --read-type image --image-sample-interval 3
CUDA_VISIBLE_DEVICES=0 python main.py /absolute/path/to/images --read-type image --remove-mask --pointcloud-downsample
python main.py --help
```

默认值集中在 `config/__init__.py`。`--remove-mask` / `--no-remove-mask` 和 `--pointcloud-downsample` / `--no-pointcloud-downsample` 可显式覆盖配置。GLB 和 PLY 分别支持置信度百分位参数。Python 调用入口为 `main.run_pipeline(...)`。

## 目录与产物

```text
app/                     FastAPI API、任务管理、隔离的流水线 worker
frontend/                Vite + 原生 JavaScript + Three.js 前端源码与锁文件
app/static/              前端构建结果（不纳入 Git）
main.py                  命令行与完整流水线入口
config/                  默认配置
pipeline/                预处理、重建、对齐阶段编排
src/                     图像与关键帧、几何、导出等实现
utils/vggt/              已纳入版本管理的 VGGT 参考实现
scripts/                 数据与导出辅助工具、结果清理工具
experiment/              历史实验脚本（保留研究记录）
tests/                   流水线回归与工作台 API 测试
data/example/            本地输入素材（不纳入 Git）
data/uploads/            工作台上传素材（不纳入 Git）
data/workbench/jobs/     请求、任务状态、worker 日志（不纳入 Git）
data/result/             重建结果（不纳入 Git）
```

工作台结果位于 `data/result/workbench/<job_id>/<task_id>/`。常见产物有 `aligned.glb`、`aligned.ply`、启用降采样时的附加文件，以及 JSON 格式的 `task_summary.log`。命令行任务默认直接写入 `data/result/<task_id>/`。成功任务会清除编号中间文件；失败或取消任务可能保留中间文件供排查。

摘要中的资源采样用于运行分析，未构成标准性能基准。系统 GPU 显示的是设备可见性，不保证模型已加载或整条任务必定成功。回环只有在有合适候选时才产生约束，单个分块不会验证回环效果。

### 输入门槛与运行证据

筛选后至少需要两个不同帧 ID；不足时在重新加载筛选图像、天空过滤及 VGGT 推理前停止，并写入原因。重建入口另按所有分块的唯一帧 ID 检查，重叠块中重复的一帧不能凑成两帧。该门槛只排除单帧输入，不保证视角差异、重建完整度或精度。

- `preprocess_evidence.json`：实际采样后扫描数、保留帧 ID、保留比例、各块规模及去重后的帧覆盖数。扫描数不是原视频总帧数；重建或对齐异常时在已有文件中补充失败阶段和原因。
- `alignment_evidence.json`：区分回环请求开启、候选/有效约束、优化尝试、采用与回退原因。单块及不足三个分块的运行不会被记为已完成回环优化。
- 成功摘要的 `parameters.run_evidence` 引用上述证据，并标明单块或跨块处理范围。这些文件在成功清理中间结果后保留；它们记录执行事实，不是几何质量评估或完整源视频帧溯源。

回环候选排除相邻块和相同帧 ID 配对；优化失败、求解抛出异常或产生非有限参数/变换时回退顺序对齐。只有优化结果确实采用才记录 `loop_optimized`，但这仍不等于已证明回环收益。

## 测试与开发

```bash
python -m pytest -q
cd frontend
npm ci
npm run check
npm run build
```

测试默认仅收集 `tests/`，避免把历史实验当作自动化测试运行。API 测试使用实际小型子进程验证状态、取消、进程清理和产物访问；流水线回归检查参数和导出路径，不下载模型或证明三维质量。前端 `npm run check` 同时执行语法检查与 Node 测试。

结题可信度补齐验收（2026-10-31）：在 `CUDA_VISIBLE_DEVICES=''` 下运行项目环境的 `python -m pytest -q`，完整回归 **112/112 通过（20.64s）**，其中本轮新增 26 项输入证据与对齐测试。覆盖不足两帧阻断、全局唯一帧检查、重叠去重计数、清理后证据保留、失败阶段、回环分支、优化失败/异常/非有限结果回退，以及默认无诊断参数调用的兼容性。未运行真实模型或 GPU 推理，当前仍缺代表性多视角与跨块质量对照。

队列版验收（2026-10-30）：完整 Python 回归 86/86 通过，前端 Node 测试 14/14 通过，生产构建成功。真实 Edge 浏览器连接 CPU 模拟服务，验证 2 个执行槽位、2 个排队位、队列位置、容量限制、取消后推进与失败重试；移动端无横向溢出、页面无运行时异常。使用单点 PLY 验证切换历史不自动下载，首次点击“查看”请求一次，刷新状态及复用当前已加载模型不重复请求。本轮不运行真实 GPU 推理，测试结果不代表多卡重建吞吐或质量基准。

首版验收（2026-10-29）：47 项 Python 测试通过，前端语法检查和生产构建通过；实际 Edge 浏览器验证服务连接、GLB / PLY 加载、日志、下载入口和移动端无横向溢出。另在 `nvidia-smi` 检查空闲情况后，以命令行前置的 `CUDA_VISIBLE_DEVICES=0` 显式指定物理 0 号卡，并核查服务进程环境确已生效，用 4 张视频抽帧跑通真实 VGGT 推理与导出，复验流水线用时约 21.1 秒，导出 268324 个点，GLB 约 4.09 MiB、PLY 约 3.84 MiB。该素材经现有关键帧策略只保留 1 帧，因此这次验证仅证明单块链路可用；不能用于证明多视角质量、跨块对齐或回环效果，也不是性能基准。后续正式实验应使用有足够视角变化的素材，并记录实际保留帧数。

每次启动 GPU 服务或命令行重建前，先运行 `nvidia-smi` 查看实时占用，再以 `CUDA_VISIBLE_DEVICES=<空闲卡编号>` 显式选择。示例中的 `0` 只作演示，不表示固定占用 0 号卡；服务启动时的选择会由任务子进程继承。

开发前端可执行 `npm run dev`，API 代理指向 `http://127.0.0.1:7860`。API 文档为 <http://127.0.0.1:7860/docs>。服务停止时，正在运行的任务会中断；重启会恢复历史记录，并将未完成任务标记为 interrupted，不会自动继续推理。

## Git 与中期版本

整理前的工作代码已提交并标记为 `archive/midterm-2026-10-29`，结题首版开发在 `feat/v1-workbench` 分支。输入素材、权重、结果、Python 缓存、前端依赖及构建产物均由忽略规则排除；原先误跟踪的字节码已取消跟踪。前端提交源码和 `package-lock.json`，由 `npm ci && npm run build` 复现。

本次清理前另存了仓库历史 bundle、工作区补丁和代码压缩包，位于服务器项目外的 `/home/zimeng/rebuild_drone_agent_backups/20261029-before-workbench/`。快照保存代码与历史，不包含大体积原始数据和历史结果。

按结题整理要求，已删除 `data/result` 与 `experiment/result` 内 27 项旧生成结果，合计 36,433,803,632 字节（约 33.93 GiB）；清单保存在外部备份目录的 `cleanup_manifest.json`。原始素材、已有权重和研究脚本保留。

以后清理先预览：

```bash
python scripts/clean_results.py
python scripts/clean_results.py --manifest /tmp/rebuild-results-inventory.json
```

确认清单后再执行 `python scripts/clean_results.py --apply`。该操作删除上述结果目录内的全部生成结果，包括新的工作台结果；不会清除上传素材或任务记录。请先下载需要保留的产物。

## 结题工作建议

沿现有技术栈收尾：准备可复现的演示素材、操作说明、环境与启动命令，并记录失败案例和适用边界。结题展示应说明排队、取消、重试和结果查看的实际流程；现有推理验证不能替代多视角质量评估，也不能将页面或队列改进表述为重建精度提升。

VGGT 上游代码与模型有各自许可；发布或使用时请核对项目及上游许可条款。当前仓库许可见 `LICENSE.txt`。
