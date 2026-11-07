import './style.css';
import { api, formatBytes, safeUrl, isActive as active, canRetry, schedulerState, submissionState, jobPhase, shouldPollFast, modelKey, previewAction, friendlyError } from './api.js';

const icons = {
  mark: '<path d="m4 19 8-15 8 15-8-5-8 5Z"/><path d="M12 4v10"/>',
  cube: '<path d="m12 3 9 5v8l-9 5-9-5V8l9-5Z"/><path d="m3 8 9 5 9-5M12 13v8M7.5 5.5l9 5"/>',
  upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v10H3V7Z"/>',
  video: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m10 9 5 3-5 3V9Z"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="m3 16 6-6 7 7 3-3 2 2"/><circle cx="16" cy="8" r="1"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>',
  arrow: '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  play: '<path d="m8 4 12 8-12 8V4Z"/>',
  refresh: '<path d="M20 7v5h-5M4 17v-5h5"/><path d="M6 6a8 8 0 0 1 13 3M5 15a8 8 0 0 0 13 3"/>',
  reset: '<path d="m3 10 9-7 9 7v11h-6v-7H9v7H3V10Z"/>',
  grid: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M15 3v18M3 9h18M3 15h18"/>',
  expand: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M3 17v4h18v-4"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 1v2m0 18v2M1 12h2m18 0h2M4 4l2 2m12 12 2 2M4 20l2-2M18 6l2-2"/>',
  chip: '<rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4"/>',
  terminal: '<path d="m4 6 6 6-6 6m9 0h7"/>'
};
const icon = (name, cls = '') => `<svg class="icon ${cls}" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.info}</svg>`;
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const $ = id => document.getElementById(id);
const labels = { queued: '排队中', running: '重建中', completed: '已完成', failed: '失败', cancelled: '已取消', interrupted: '已中断' };
const stamp = value => {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).format(date);
};
let savedId = null;
try { savedId = localStorage.getItem('aeroscape.selected-job'); } catch { /* Storage may be unavailable. */ }
const state = { jobs: [], selectedId: savedId, selected: null, system: null, items: [], files: [], source: 'upload', uploaded: null, busy: false, busyKind: null, cancellingIds: new Set(), retriedIds: new Map(), connected: false, viewer: null, modelKey: null, loadedKey: null, requestedKey: null, previewUrl: null, poll: null, refreshing: false, jobRevision: 0 };

document.querySelector('#app').innerHTML = `
  <header class="topbar">
    <a class="brand" href="/" aria-label="航境 AEROSCAPE 首页"><span class="brand-symbol">${icon('mark')}</span><span class="brand-name">航境 <span class="brand-dot">·</span> <span class="wordmark">AEROSCAPE</span></span></a>
    <div class="header-section"><span class="header-divider"></span><span>三维重建工作台</span></div>
    <div class="system-pill" id="system-pill" role="status"><span class="status-dot waiting"></span><span id="connection-label">正在连接服务</span></div>
    <button class="icon-button" id="refresh-all" aria-label="刷新任务与系统状态" title="刷新任务与系统状态">${icon('refresh')}</button>
  </header>
  <main class="workspace">
    <aside class="configuration" aria-label="新建重建任务">
      <div class="panel-heading"><div><p class="eyebrow">NEW RECONSTRUCTION</p><h1>从影像，抵达空间。</h1></div><span class="heading-mark">${icon('cube')}</span></div>
      <form id="task-form">
        <section class="form-section"><div class="section-label"><span class="step">01</span><h2>导入影像</h2></div>
          <div class="source-tabs" role="tablist" aria-label="影像来源">
            <button type="button" id="tab-upload" role="tab" aria-controls="source-upload" aria-selected="true" tabindex="0" data-source="upload">本地上传</button>
            <button type="button" id="tab-server" role="tab" aria-controls="source-server" aria-selected="false" tabindex="-1" data-source="server">服务器示例</button>
            <button type="button" id="tab-path" role="tab" aria-controls="source-path" aria-selected="false" tabindex="-1" data-source="path">路径输入</button>
          </div>
          <div id="source-upload" role="tabpanel" aria-labelledby="tab-upload">
            <label class="dropzone" id="dropzone" for="file-input"><span class="upload-symbol">${icon('upload')}</span><strong>拖入影像，或<span>选择文件</span></strong><span class="drop-caption">一个视频 / 多张连续航拍照片</span><span class="file-formats">MP4 · MOV · AVI · MKV · JPG · PNG · WEBP</span><input id="file-input" type="file" multiple accept=".mp4,.mov,.avi,.mkv,.m4v,.webm,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff" aria-label="选择一个视频或多张图片" /></label>
            <div id="file-selection" class="file-selection" hidden></div>
          </div>
          <div id="source-server" role="tabpanel" aria-labelledby="tab-server" hidden><label for="server-input">选择已有影像</label><select id="server-input"><option value="">正在读取服务器示例…</option></select><p class="field-hint" id="server-hint">使用服务器上已有的数据，无需上传。</p><button type="button" class="text-button" id="refresh-inputs">${icon('refresh')} 刷新示例</button></div>
          <div id="source-path" role="tabpanel" aria-labelledby="tab-path" hidden><label for="input-path">服务器上的绝对路径</label><input id="input-path" type="text" placeholder="/data/flight/video.mp4" autocomplete="off" spellcheck="false" /><label for="read-type">影像类型</label><select id="read-type"><option value="video">视频文件</option><option value="image">图像目录</option></select><p class="field-hint">填写运行服务的机器上的路径。</p></div>
          <div class="inline-error" id="input-error" role="alert" hidden></div>
        </section>
        <section class="form-section"><div class="section-label"><span class="step">02</span><h2>重建设置</h2></div>
          <label for="task-name">任务名称 <span class="optional">可选</span></label><input id="task-name" maxlength="100" placeholder="例如：园区航测 · 西区" autocomplete="off" />
          <div class="field-heading"><label for="sample-interval">图像采样间隔</label><span class="unit-label">张</span></div>
          <div class="number-control"><button type="button" id="interval-minus" aria-label="减小采样间隔">−</button><input id="sample-interval" type="number" min="1" max="1000" step="1" value="3" required aria-describedby="interval-hint" /><button type="button" id="interval-plus" aria-label="增大采样间隔">+</button></div>
          <p class="field-hint" id="interval-hint">仅图像目录生效，每隔 3 张采样；视频按 3 帧/秒提取。</p>
          <details class="advanced"><summary>高级参数 <span>精细控制</span></summary><div class="advanced-content">
            <label class="switch-row" for="remove-mask"><span>过滤天空<small>使用 SegFormer 天空掩码</small></span><input id="remove-mask" type="checkbox" role="switch" /><span class="switch-track"></span></label>
            <label class="switch-row" for="downsample"><span>点云降采样<small>降低产物体积与查看负担</small></span><input id="downsample" type="checkbox" role="switch" /><span class="switch-track"></span></label>
            <div class="field-heading"><label for="confidence">置信度过滤百分位</label><output id="confidence-value" for="confidence">0</output></div><input id="confidence" type="range" min="0" max="100" step="1" value="0" /><p class="field-hint">0 表示不过滤；数值越大，保留点越少。</p>
          </div></details>
        </section>
        <div class="submit-area"><div id="queue-capacity" class="queue-capacity" role="status" aria-live="polite">正在读取调度容量…</div><div id="submit-error" class="inline-error" role="alert" hidden></div><button type="submit" class="primary-button" id="start-button">${icon('play')}<span>开始重建</span>${icon('arrow')}</button><p id="submit-hint">影像将用于本机重建流水线</p></div>
      </form>
      <div class="compute-info"><span class="compute-icon">${icon('chip')}</span><div><strong id="gpu-name">计算资源待连接</strong><span id="disk-info">连接后显示 GPU 与可用磁盘</span></div></div>
    </aside>
    <section class="work-area" aria-label="重建结果与任务">
      <div class="work-heading"><div><p class="eyebrow">SPATIAL WORKSPACE</p><h2>空间视图 <span class="subtle-divider">/</span> <span id="selected-name">等待首个重建任务</span></h2></div><span class="view-tag">${icon('cube')} 3D VIEWER</span></div>
      <div class="connection-error" id="connection-error" role="alert" hidden></div>
      <section class="viewer-shell" id="viewer-shell" aria-label="三维结果查看器">
        <div class="viewer-top"><div class="viewer-status"><span class="status-dot"></span><span id="viewer-caption">工作空间已就绪</span></div><span class="projection">透视视图</span></div>
        <div class="canvas-host" id="canvas-host"></div>
        <div class="viewport-empty" id="viewport-empty"><span class="empty-cube">${icon('cube')}</span><p class="empty-kicker">YOUR NEXT PERSPECTIVE</p><h3 id="empty-title">让每一帧，拥有新的维度</h3><p id="empty-message">导入航拍影像，开始一次三维重建。<br />生成的模型将在这里呈现。</p><button type="button" class="viewport-cta" id="viewport-upload">导入第一组影像 ${icon('arrow')}</button><span class="grid-note">参考网格 · 当前无重建结果</span></div>
        <div class="viewer-notice" id="viewer-notice" role="status" hidden></div>
        <div class="viewer-tools" role="toolbar" aria-label="视图控制"><button type="button" id="reset-view" title="重置视角" aria-label="重置视角">${icon('reset')}</button><span></span><button type="button" id="toggle-grid" title="显示参考网格" aria-label="显示参考网格" aria-pressed="true">${icon('grid')}</button><button type="button" id="toggle-background" title="切换浅色背景" aria-label="切换浅色背景" aria-pressed="false">${icon('sun')}</button><span></span><button type="button" id="fullscreen" title="全屏查看" aria-label="全屏查看">${icon('expand')}</button></div>
        <div class="viewer-bottom"><span class="navigation-hint"><span>拖动</span> 旋转 <b>·</b> <span>滚轮</span> 缩放 <b>·</b> <span>右键</span> 平移</span><label class="point-control" for="point-size">点大小<input id="point-size" type="range" min="0.001" max="0.1" value="0.025" step="0.001" /></label><span class="axis-symbol" aria-hidden="true"><i>Z</i><i>Y</i><i>X</i></span></div>
      </section>
      <section class="task-card" aria-label="当前任务详情"><div class="task-detail-top"><div class="current-task-title"><span class="section-tick"></span><h3>当前任务</h3><span id="job-badge" role="status" aria-live="polite" class="badge neutral">未选择</span></div><div class="detail-actions"><button type="button" class="text-button" id="refresh-job" disabled>${icon('refresh')}刷新</button><button type="button" class="retry-button" id="retry-job" hidden>${icon('refresh')}<span>重新提交</span></button><button type="button" class="cancel-button" id="cancel-job" hidden>${icon('close')}取消任务</button></div></div><div id="job-detail" class="job-detail"><p class="muted">任务启动后，可在这里查看实时阶段、日志与输出文件。</p></div><div id="job-error" class="inline-error job-error" role="alert" hidden></div><div id="outputs" class="outputs" hidden></div><details id="logs-panel" class="logs-panel"><summary><span>${icon('terminal')}运行日志<span class="log-note" id="log-note">选择任务后可查看</span></span><span class="log-expand">展开</span></summary><pre id="job-logs" tabindex="0" aria-label="任务运行日志">尚未选择任务</pre></details></section>
      <section class="history-section" aria-label="任务历史"><div class="history-heading"><div><h3>任务记录</h3><span id="history-count">0 个任务</span></div><button type="button" class="text-button" id="refresh-history">${icon('refresh')}刷新记录</button></div><div id="history-error" class="inline-error" role="alert" hidden></div><div class="job-list" id="job-list"><div class="history-empty">${icon('clock')}<span>正在读取任务记录…</span></div></div></section>
      <footer class="workspace-footer"><span>AEROSCAPE <b>/</b> 影像与空间之间</span><span id="version-label">本地三维重建工作台</span></footer>
    </section>
  </main><div class="toast" id="toast" role="status" hidden></div>`;

function showError(id, message) { $(id).textContent = message || ''; $(id).hidden = !message; }
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toast.timer); toast.timer = setTimeout(() => { $('toast').hidden = true; }, 5000); }
function selectSource(source, focus = false) {
  if (state.busy) return;
  state.source = source;
  document.querySelectorAll('[data-source]').forEach(button => {
    const selected = button.dataset.source === source;
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
    $(`source-${button.dataset.source}`).hidden = !selected;
  });
  showError('input-error', ''); updateStart();
}
function sourceReady() {
  if (state.source === 'upload') return state.files.length > 0;
  if (state.source === 'server') return $('server-input').value !== '';
  return $('input-path').value.trim().length > 0;
}
function updateStart() {
  const submission = submissionState({ ...state, ready: sourceReady() });
  $('start-button').disabled = submission.disabled;
  $('start-button').classList.toggle('loading', state.busy);
  $('start-button').querySelector('span').textContent = state.busyKind === 'retry' ? '正在重新提交…' : submission.label;
  const configurationHint = !submission.schedulerReady ? `未配置任务设备，暂不可提交；请联系管理员检查调度配置。${submission.configurationError ? ` 详情：${submission.configurationError}` : ''}` : '';
  $('submit-hint').textContent = state.busy ? '正在创建任务，请勿重复提交' : !state.connected ? '等待连接重建服务' : configurationHint || (submission.full ? '队列已满，可继续填写；空位释放后即可提交' : submission.willQueue ? '提交后按队列顺序自动开始，可随时取消排队' : '影像将用于本机重建流水线');
  $('queue-capacity').textContent = configurationHint || (state.system ? `运行 ${submission.running}/${submission.concurrent} · 排队 ${submission.queued}/${submission.maxQueue ?? '未知'} · ${submission.slots == null ? '等待容量更新' : `剩余 ${submission.slots} 个排队位`}` : '正在读取调度容量…');
  $('queue-capacity').classList.toggle('is-full', submission.full || !submission.schedulerReady);
  document.querySelectorAll('.source-tabs button, #file-input, #input-path, #read-type, #server-input, #task-name, #sample-interval, #interval-minus, #interval-plus, #remove-mask, #downsample, #confidence, #clear-files, #viewport-upload').forEach(element => { element.disabled = state.busy; });
  $('retry-job').disabled = state.busy || !state.connected || !submission.schedulerReady || submission.full || state.retriedIds.has(state.selectedId);
  $('retry-job').querySelector('span').textContent = state.busyKind === 'retry' && state.retryingId === state.selectedId ? '正在重新提交…' : state.retriedIds.has(state.selectedId) ? '已重新提交' : '重新提交';
  $('retry-job').title = state.retriedIds.has(state.selectedId) ? `已创建新任务 ${state.retriedIds.get(state.selectedId)}` : configurationHint || (submission.full ? '等待队列空位后可重新提交' : '复用此任务的输入与参数，创建一条新任务记录');
}

function setConnection(connected, error = '') {
  state.connected = connected;
  $('connection-label').textContent = connected ? '服务已连接' : '服务连接中断';
  $('system-pill').classList.toggle('offline', !connected);
  $('system-pill').querySelector('.status-dot').classList.toggle('waiting', !connected);
  showError('connection-error', error ? `${error} 当前显示上次获取的状态。可点击右上角刷新重连。` : '');
  updateStart();
}

async function loadSystem() {
  try {
    const system = await api('/system'); state.system = system;
    $('gpu-name').textContent = system.gpu?.available ? system.gpu.name || 'GPU 可用' : 'GPU 不可用';
    $('disk-info').textContent = `${system.gpu?.memory_total_gb != null ? `显存 ${Number(system.gpu.memory_total_gb).toFixed(1)} GB · ` : ''}${system.disk?.free_gb != null ? `磁盘可用 ${Number(system.disk.free_gb).toFixed(1)} GB` : '磁盘空间未知'}`;
    $('version-label').textContent = system.version ? `版本 ${system.version}` : '本地三维重建工作台';
    setConnection(true);
  } catch (error) { setConnection(false, error.message); }
}

async function loadInputs() {
  $('refresh-inputs').disabled = true;
  try {
    const data = await api('/inputs'); state.items = data.items || [];
    const old = $('server-input').value;
    $('server-input').innerHTML = `<option value="">${state.items.length ? '请选择服务器影像' : '暂无可用服务器示例'}</option>${state.items.map((item, index) => `<option value="${index}">${esc(item.name)} · ${item.type === 'video' ? '视频' : '图像'}</option>`).join('')}`;
    if (state.items[Number(old)] && old !== '') $('server-input').value = old;
    $('server-hint').textContent = state.items.length ? '使用服务器上已有的数据，无需上传。' : '可上传本地影像，或输入服务器路径。';
  } catch (error) {
    $('server-input').innerHTML = '<option value="">无法读取服务器示例</option>';
    $('server-hint').textContent = error.message;
  } finally { $('refresh-inputs').disabled = false; updateStart(); }
}

function acceptFiles(files) {
  if (state.busy) return;
  const list = Array.from(files);
  const videos = /\.(mp4|mov|avi|mkv|m4v|webm)$/i;
  const images = /\.(jpe?g|png|webp|bmp|tiff?)$/i;
  let error = '';
  if (!list.length) return;
  if (list.some(file => !videos.test(file.name) && !images.test(file.name))) error = '包含不支持的文件。请选择视频或 JPG、PNG、WEBP、BMP、TIFF 图片。';
  else if (list.some(file => videos.test(file.name)) && (list.length !== 1 || list.some(file => images.test(file.name)))) error = '请仅选择一个视频，或选择多张图片；视频与图片不能混合上传。';
  else if (list.every(file => images.test(file.name)) && list.length < 2) error = '图像重建至少需要 2 张照片，请重新选择多张连续航拍照片。';
  else if (list.some(file => file.size === 0)) error = '包含空文件，请移除后重新选择。';
  state.files = error ? [] : list; state.uploaded = null;
  showError('input-error', error);
  $('file-selection').hidden = !state.files.length;
  $('dropzone').classList.toggle('has-files', Boolean(state.files.length));
  if (state.files.length) {
    $('file-selection').innerHTML = `<div class="file-summary">${icon(videos.test(list[0].name) ? 'video' : 'image')}<strong>${list.length === 1 ? esc(list[0].name) : `${list.length} 张航拍照片`}</strong><button type="button" class="icon-button" id="clear-files" aria-label="移除选中的文件">${icon('close')}</button></div><div class="file-total">${formatBytes(list.reduce((sum, file) => sum + file.size, 0))}<span>${list.length > 1 ? esc(list.slice(0, 3).map(file => file.name).join('、')) + (list.length > 3 ? '…' : '') : '影像已就绪，开始重建时上传'}</span></div>`;
    $('clear-files').addEventListener('click', () => {
      state.files = []; state.uploaded = null; $('file-input').value = ''; $('file-selection').hidden = true; $('dropzone').classList.remove('has-files'); updateStart();
    });
  }
  updateStart();
}

function renderJobs() {
  $('history-count').textContent = `${state.jobs.length} 个任务`;
  const signature = JSON.stringify(state.jobs.map(job => [job.id, job.name, job.status, job.stage, job.created_at, job.started_at, job.queue_position, job.gpu_device])) + state.selectedId;
  if (renderJobs.signature === signature) return;
  renderJobs.signature = signature;
  $('job-list').innerHTML = state.jobs.length ? state.jobs.map(job => `<button type="button" class="job-row ${job.id === state.selectedId ? 'selected' : ''}" data-job-id="${esc(job.id)}" aria-pressed="${job.id === state.selectedId}"><span class="job-icon ${esc(job.status)}">${icon(job.status === 'completed' ? 'check' : active(job) ? 'cube' : job.status === 'failed' || job.status === 'interrupted' ? 'info' : 'clock')}</span><span class="job-identity"><strong>${esc(job.name || job.id)}</strong><small>${esc(job.read_type === 'image' ? '图像序列' : '视频影像')}<b>·</b>${esc(stamp(job.created_at))}</small><small class="job-scheduling">${esc(jobPhase(job))}${job.status === 'running' ? ` · GPU ${esc(job.gpu_device ?? '分配中')}` : ''}</small></span><span class="job-stage">${esc(active(job) ? '' : jobPhase(job))}</span><span class="badge ${esc(job.status)}">${labels[job.status] || esc(job.status)}</span>${icon('chevron')}</button>`).join('') : `<div class="history-empty">${icon('clock')}<span>还没有任务。导入第一组影像，开始建立你的空间档案。</span></div>`;
}

function renderSelected() {
  const job = state.selected;
  $('selected-name').textContent = job?.name || (job ? job.id : '等待首个重建任务');
  $('refresh-job').disabled = !job;
  $('cancel-job').hidden = !active(job);
  $('cancel-job').disabled = state.cancellingIds.has(job?.id);
  $('cancel-job').innerHTML = `${icon('close')}${state.cancellingIds.has(job?.id) ? '正在取消…' : job?.status === 'queued' ? '取消排队' : '取消任务'}`;
  $('retry-job').hidden = !canRetry(job);
  $('job-badge').className = `badge ${job?.status || 'neutral'}`;
  $('job-badge').textContent = job ? labels[job.status] || job.status : '未选择';
  showError('job-error', job?.error ? friendlyError(job.error) : '');
  if (!job) { updateStart(); return; }
  const statusSentence = active(job) ? jobPhase(job) : job.status === 'completed' ? '重建完成，点击下方“查看”后加载模型。' : job.status === 'failed' ? '任务执行失败，可检查日志后重新提交。' : job.status === 'interrupted' ? '任务已中断，可复用原输入与参数重新提交。' : '任务已取消，可重新提交为新任务。';
  $('job-detail').innerHTML = `<div class="stage-line ${job.status === 'running' && job.started_at ? 'is-active' : ''}"><span class="stage-dot"></span><strong>${esc(statusSentence)}</strong></div><div class="job-metadata"><span>创建 <b>${esc(stamp(job.created_at))}</b></span><span>开始 <b>${esc(job.status === 'queued' ? '尚未开始' : stamp(job.started_at))}</b></span><span>结束 <b>${esc(stamp(job.finished_at))}</b></span>${job.status === 'running' ? `<span>运行设备 <b>GPU ${esc(job.gpu_device ?? '分配中')}</b></span>` : ''}${job.retry_of ? `<span>重试来源 <b>${esc(job.retry_of)}</b></span>` : ''}</div><p class="input-reference" title="${esc(job.input_path)}">${icon('folder')}<span>${esc(job.input_path)}</span></p>`;
  const outputs = (job.outputs || []).filter(output => safeUrl(output.url));
  const outputSignature = JSON.stringify(outputs) + job.id;
  if (renderSelected.outputSignature !== outputSignature) {
    renderSelected.outputSignature = outputSignature;
    $('outputs').hidden = outputs.length === 0;
    $('outputs').innerHTML = outputs.map((output, index) => `<div class="output-file">${icon(/\.(glb|ply)$/i.test(output.name) ? 'cube' : 'folder')}<span><strong>${esc(output.name)}</strong><small>${formatBytes(output.size_bytes)}</small></span>${/\.(glb|ply)$/i.test(output.name) ? `<button type="button" class="text-button view-output" data-output-index="${index}">查看</button>` : ''}<a href="${esc(safeUrl(output.url))}" download="${esc(output.name)}" class="download-button" aria-label="下载 ${esc(output.name)}" title="下载 ${esc(output.name)}">${icon('download')}</a></div>`).join('');
    $('outputs').querySelectorAll('.view-output').forEach(button => button.addEventListener('click', () => {
      const output = outputs[Number(button.dataset.outputIndex)];
      state.previewUrl = output.url;
      state.requestedKey = modelKey(job.id, output);
      loadModel(output);
    }));
  }
  if (job.summary && Object.keys(job.summary).length) {
    const details = document.createElement('details'); details.className = 'summary-data';
    const summary = document.createElement('summary'); summary.textContent = '重建摘要';
    const pre = document.createElement('pre'); pre.textContent = JSON.stringify(job.summary, null, 2);
    details.append(summary, pre); $('job-detail').appendChild(details);
  }
  const models = outputs.filter(output => /\.(glb|ply)$/i.test(output.name));
  const preview = models.find(output => modelKey(job.id, output) === state.requestedKey);
  const showing = Boolean(preview && state.modelKey === state.requestedKey);
  $('canvas-host').hidden = !showing;
  $('viewport-empty').hidden = showing;
  if (!showing) {
    $('viewport-upload').hidden = active(job) || job.status === 'completed' || models.length > 0;
    $('empty-title').textContent = models.length ? '模型已就绪，按需查看' : job.status === 'queued' ? jobPhase(job) : job.status === 'running' ? '等待重建结果' : job.status === 'completed' ? '此任务暂无可预览模型' : '本次重建尚未完成';
    $('empty-message').textContent = models.length ? `下方 ${models.length} 个模型可预览（${models.map(output => formatBytes(output.size_bytes)).join(' / ')}）。点击“查看”才会传输模型；切换任务和刷新状态不会自动下载。` : job.status === 'queued' ? '任务尚未开始；轮到此任务后自动运行，也可取消排队。' : job.status === 'running' ? '流水线正在处理影像。输出就绪后，可在下方点击“查看”。' : '查看下方任务详情、日志与输出文件。';
    $('viewer-caption').textContent = models.length ? '模型尚未请求 · 点击“查看”加载' : job.status === 'queued' ? '排队等待中 · 尚未开始' : '等待可预览模型';
  }
  updatePreviewButtons(outputs);
  $('log-note').textContent = active(job) ? '展开后自动更新' : '查看实际运行输出';
  updateStart();
}

function updatePreviewButtons(outputs = (state.selected?.outputs || []).filter(output => safeUrl(output.url))) {
  $('outputs').querySelectorAll('.view-output').forEach(button => {
    const output = outputs[Number(button.dataset.outputIndex)];
    if (!output) return;
    const key = modelKey(state.selectedId, output);
    const loading = state.modelKey === key && state.loadedKey !== key;
    button.textContent = state.loadedKey === key ? '查看（已加载）' : loading ? '加载中…' : '查看';
    button.disabled = loading;
    button.title = `预览将传输 ${formatBytes(output.size_bytes)}；只在点击后加载`;
  });
}

async function loadModel(output) {
  const key = modelKey(state.selectedId, output);
  const action = previewAction({ key, requestedKey: state.requestedKey, loadingKey: state.modelKey, loadedKey: state.loadedKey, ready: Boolean(state.viewer) });
  if (action === 'idle') return;
  $('canvas-host').hidden = false;
  state.viewer.resize();
  $('viewport-empty').hidden = true;
  if (action === 'wait') {
    $('viewer-notice').textContent = `正在加载 ${output.name}（${formatBytes(output.size_bytes)}）…`; $('viewer-notice').hidden = false;
    $('viewer-caption').textContent = output.name;
    return;
  }
  if (action === 'reuse') {
    $('viewer-notice').hidden = true;
    $('viewer-caption').textContent = `${output.name} · 已加载，未重复下载`;
    return;
  }
  state.modelKey = key; state.loadedKey = null;
  $('viewer-notice').textContent = `正在加载 ${output.name}（${formatBytes(output.size_bytes)}）…`; $('viewer-notice').hidden = false;
  $('viewer-caption').textContent = output.name;
  updatePreviewButtons();
  try {
    const loaded = await state.viewer.load(safeUrl(output.url), /\.ply$/i.test(output.name) ? 'ply' : 'glb');
    if (state.modelKey !== key || !loaded) return;
    state.loadedKey = key;
    if (state.requestedKey !== key) return;
    $('viewer-notice').hidden = true;
    $('viewer-caption').textContent = `${output.name} · 实际重建结果`;
    updatePreviewButtons();
  } catch (error) {
    if (state.modelKey !== key) return;
    state.modelKey = null;
    if (state.requestedKey !== key) return;
    $('viewer-notice').textContent = `模型加载失败：${error.message} 可在下方重新点击“查看”，或下载文件。`;
    $('viewer-notice').hidden = false;
    updatePreviewButtons();
  }
}

async function loadLogs() {
  const id = state.selectedId;
  if (!id || !$('logs-panel').open) return;
  try {
    const data = await api(`/jobs/${encodeURIComponent(id)}/logs`);
    if (id !== state.selectedId) return;
    const pre = $('job-logs');
    const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 48;
    pre.textContent = data.lines?.length ? data.lines.join('\n') : '暂无日志，等待流水线输出。';
    if (atBottom) pre.scrollTop = pre.scrollHeight;
    $('log-note').textContent = active(state.selected) ? '实时日志' : '运行日志';
  } catch (error) { if (id === state.selectedId) $('log-note').textContent = `日志读取失败：${error.message}`; }
}

async function selectJob(id) {
  const changed = state.selectedId !== id;
  state.selectedId = id;
  try { localStorage.setItem('aeroscape.selected-job', id); } catch { /* Optional persistence. */ }
  state.selected = state.jobs.find(job => job.id === id) || null;
  if (changed) { state.requestedKey = null; state.previewUrl = null; $('canvas-host').hidden = true; }
  $('viewer-notice').hidden = true; $('job-logs').textContent = '正在读取日志…';
  if (state.selected) renderSelected();
  renderJobs();
  const revision = state.jobRevision;
  try {
    const job = await api(`/jobs/${encodeURIComponent(id)}`);
    if (state.selectedId !== id || state.jobRevision !== revision) return;
    state.selected = job; renderSelected(); await loadLogs();
  } catch (error) { if (state.selectedId === id) showError('job-error', error.message); }
}

async function loadJobs() {
  const revision = state.jobRevision;
  const data = await api('/jobs');
  if (state.jobRevision !== revision) return;
  state.jobs = (data.jobs || []).slice().sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  showError('history-error', ''); renderJobs();
  if (!state.selectedId || !state.jobs.some(job => job.id === state.selectedId)) {
    const next = state.jobs.find(active) || state.jobs[0];
    if (next) await selectJob(next.id);
    else { state.selectedId = null; state.selected = null; renderSelected(); }
  } else {
    const id = state.selectedId;
    const job = await api(`/jobs/${encodeURIComponent(id)}`);
    if (state.selectedId === id && state.jobRevision === revision) { state.selected = job; renderSelected(); }
  }
  updateStart();
}

function schedulePoll() {
  clearTimeout(state.poll);
  if (document.hidden) return;
  state.poll = setTimeout(refresh, shouldPollFast(state.system, state.jobs, state.connected) && state.connected ? 3500 : 12000);
}
async function refresh() {
  if (state.refreshing) return;
  state.refreshing = true; $('refresh-all').classList.add('spinning');
  try {
    await loadSystem();
    await loadJobs();
    await loadLogs();
  } catch (error) {
    showError('history-error', error.message);
    setConnection(false, error.message);
  } finally { state.refreshing = false; $('refresh-all').classList.remove('spinning'); schedulePoll(); }
}

function rememberJob(job) {
  state.jobRevision++;
  state.jobs = [job, ...state.jobs.filter(item => item.id !== job.id)];
  if (state.system?.scheduler) {
    const running = state.jobs.filter(item => item.status === 'running');
    const queued = state.jobs.filter(item => item.status === 'queued');
    state.system.scheduler = { ...state.system.scheduler, running_count: running.length, queued_count: queued.length, available_queue_slots: Math.max(0, state.system.scheduler.max_queue - queued.length) };
    state.system.active_job_ids = running.map(item => item.id);
    state.system.active_job_id = running[0]?.id || null;
  }
  renderJobs(); updateStart();
}

$('task-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (state.busy || $('start-button').disabled || !$('task-form').reportValidity()) return;
  showError('submit-error', ''); state.busy = true; state.busyKind = 'create'; updateStart();
  try {
    let input;
    if (state.source === 'upload') {
      if (!state.uploaded) {
        const form = new FormData(); state.files.forEach(file => form.append('files', file));
        state.uploaded = await api('/uploads', { method: 'POST', body: form });
      }
      input = state.uploaded;
    } else if (state.source === 'server') {
      const selected = state.items[Number($('server-input').value)];
      input = { input_path: selected.path, read_type: selected.type };
    } else input = { input_path: $('input-path').value.trim(), read_type: $('read-type').value };
    const job = await api('/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ input_path: input.input_path, read_type: input.read_type, name: $('task-name').value.trim() || undefined, image_sample_interval: Number($('sample-interval').value), remove_mask_enable: $('remove-mask').checked, confidence_percentile: Number($('confidence').value), downsample_enable: $('downsample').checked }) });
    rememberJob(job);
    await selectJob(job.id);
    toast(job.status === 'queued' ? `任务已加入队列：${jobPhase(job)}。` : '任务已创建，流水线状态将在这里实时更新。');
    await loadSystem();
  } catch (error) { showError('submit-error', error.message); if (error.status === 409) await loadSystem(); }
  finally { state.busy = false; state.busyKind = null; updateStart(); schedulePoll(); }
});

$('cancel-job').addEventListener('click', async () => {
  const id = state.selectedId;
  if (state.cancellingIds.has(id) || !active(state.selected)) return;
  state.cancellingIds.add(id); renderSelected();
  try {
    const job = await api(`/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
    rememberJob(job);
    if (state.selectedId === id) { state.selected = job; renderSelected(); }
    toast(active(job) ? '取消请求已发送，等待任务停止。' : '任务已取消。');
    await loadSystem();
  } catch (error) { if (state.selectedId === id) showError('job-error', error.message); else toast(`取消任务未完成：${error.message}`); }
  finally {
    state.cancellingIds.delete(id);
    $('cancel-job').disabled = state.cancellingIds.has(state.selectedId);
    $('cancel-job').innerHTML = `${icon('close')}${state.cancellingIds.has(state.selectedId) ? '正在取消…' : state.selected?.status === 'queued' ? '取消排队' : '取消任务'}`;
    updateStart(); schedulePoll();
  }
});

$('retry-job').addEventListener('click', async () => {
  const original = state.selected;
  const id = original?.id;
  const submission = submissionState({ ...state, ready: true });
  if (submission.disabled || !canRetry(original) || state.retriedIds.has(id)) return;
  state.busy = true; state.busyKind = 'retry'; state.retryingId = id;
  showError('job-error', ''); updateStart();
  try {
    const job = await api(`/jobs/${encodeURIComponent(id)}/retry`, { method: 'POST' });
    state.retriedIds.set(id, job.id);
    rememberJob(job);
    if (state.selectedId === id) await selectJob(job.id);
    toast(`已重新提交为新任务${job.status === 'queued' ? `：${jobPhase(job)}` : '，开始重建'}。原记录保留。`);
    await loadSystem();
  } catch (error) {
    if (state.selectedId === id) showError('job-error', error.message); else toast(`重新提交未完成：${error.message}`);
    if (error.status === 409) await loadSystem();
  } finally { state.busy = false; state.busyKind = null; state.retryingId = null; updateStart(); schedulePoll(); }
});

document.querySelectorAll('[data-source]').forEach((button, index, buttons) => {
  button.addEventListener('click', () => selectSource(button.dataset.source));
  button.addEventListener('keydown', event => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
      selectSource(buttons[next].dataset.source, true);
    }
  });
});
$('file-input').addEventListener('change', event => acceptFiles(event.target.files));
['dragenter', 'dragover'].forEach(type => $('dropzone').addEventListener(type, event => { event.preventDefault(); if (!state.busy) $('dropzone').classList.add('dragover'); }));
['dragleave', 'drop'].forEach(type => $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.remove('dragover'); }));
$('dropzone').addEventListener('drop', event => acceptFiles(event.dataTransfer.files));
$('server-input').addEventListener('change', updateStart); $('input-path').addEventListener('input', updateStart);
$('refresh-inputs').addEventListener('click', loadInputs);
$('confidence').addEventListener('input', event => { $('confidence-value').value = event.target.value; });
function intervalChange(delta = 0) {
  const input = $('sample-interval');
  if (delta) input.value = Math.min(1000, Math.max(1, (Number(input.value) || 1) + delta));
  $('interval-hint').textContent = `仅图像目录生效，每隔 ${input.value || '—'} 张采样；视频按 3 帧/秒提取。`;
}
$('sample-interval').addEventListener('input', () => intervalChange());
$('interval-minus').addEventListener('click', () => intervalChange(-1)); $('interval-plus').addEventListener('click', () => intervalChange(1));
$('job-list').addEventListener('click', event => { const button = event.target.closest('[data-job-id]'); if (button) selectJob(button.dataset.jobId); });
$('refresh-all').addEventListener('click', refresh); $('refresh-history').addEventListener('click', refresh); $('refresh-job').addEventListener('click', refresh);
$('logs-panel').addEventListener('toggle', loadLogs);
$('viewport-upload').addEventListener('click', () => { selectSource('upload'); $('file-input').click(); });
$('reset-view').addEventListener('click', () => state.viewer?.reset());
$('toggle-grid').addEventListener('click', event => { const button = event.currentTarget; const visible = button.getAttribute('aria-pressed') !== 'true'; button.setAttribute('aria-pressed', String(visible)); state.viewer?.setGrid(visible); });
$('toggle-background').addEventListener('click', event => { const button = event.currentTarget; const light = button.getAttribute('aria-pressed') !== 'true'; button.setAttribute('aria-pressed', String(light)); state.viewer?.setLight(light); $('viewer-shell').classList.toggle('light-view', light); });
$('point-size').addEventListener('input', event => state.viewer?.setPointSize(event.target.value));
$('fullscreen').addEventListener('click', async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else if ($('viewer-shell').requestFullscreen) await $('viewer-shell').requestFullscreen(); else toast('当前浏览器不支持全屏显示。'); }
  catch (error) { toast(`无法进入全屏：${error.message}`); }
});
document.addEventListener('fullscreenchange', () => { const full = Boolean(document.fullscreenElement); $('fullscreen').setAttribute('aria-label', full ? '退出全屏' : '全屏查看'); $('fullscreen').setAttribute('title', full ? '退出全屏' : '全屏查看'); });
document.addEventListener('visibilitychange', () => { if (document.hidden) clearTimeout(state.poll); else refresh(); });
window.addEventListener('online', refresh);
window.addEventListener('offline', () => { setConnection(false, '网络已断开。'); schedulePoll(); });
window.addEventListener('pagehide', event => { clearTimeout(state.poll); if (!event.persisted) state.viewer?.dispose(); });
window.addEventListener('pageshow', event => { if (event.persisted) refresh(); });
async function initializeViewer() {
  try {
    const { SceneViewer } = await import('./viewer.js');
    state.viewer = new SceneViewer($('canvas-host'), message => { $('viewer-notice').textContent = message; $('viewer-notice').hidden = false; });
    if (state.selected) {
      renderSelected();
      const requested = state.selected.outputs?.find(output => modelKey(state.selectedId, output) === state.requestedKey);
      if (requested) loadModel(requested);
    }
  } catch {
    $('viewer-notice').textContent = '无法启用 3D 视口，请使用支持 WebGL 的浏览器。任务提交与文件下载仍可使用。';
    $('viewer-notice').hidden = false;
    document.querySelectorAll('.viewer-tools button, #point-size').forEach(button => { button.disabled = true; });
  }
}
updateStart();
initializeViewer();
loadInputs();
refresh();
