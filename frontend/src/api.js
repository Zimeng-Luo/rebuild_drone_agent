export function backendDetail(detail) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map(item => typeof item === 'string' ? item : `${item.loc?.join('.') ? `${item.loc.join('.')}: ` : ''}${item.msg || JSON.stringify(item)}`).join('；');
  return detail == null ? '' : JSON.stringify(detail);
}

export function friendlyError(detail, status = 0) {
  const text = backendDetail(detail);
  let message = '请求未完成，请查看详情后重试。';
  if (/queue.*full|full.*queue|队列.*满/i.test(text)) message = '等待队列已满，请等候空位后重新提交。';
  else if (/gpu.*config|config.*gpu|explicit.*gpu|gpu.*device.*required/i.test(text)) message = '服务尚未配置可用的 GPU 任务设备，请联系管理员检查调度配置。';
  else if (/retry|failed.*cancelled.*interrupted/i.test(text) && status === 409) message = '此任务当前不能重新提交，请刷新状态后再试。';
  else if (/not found|does not exist|no such file/i.test(text) || status === 404) message = '任务、输入路径或文件不存在，请检查后重试。';
  else if (/shutting down|shutdown|closed/i.test(text)) message = '服务正在停止，请稍后刷新连接。';
  else if (/already running/i.test(text)) message = '当前服务暂不接受新任务，请等待后重试。';
  else if (status === 422 || status === 400) message = '输入参数未通过检查，请核对影像路径和重建设置。';
  else if (status === 409) message = '任务状态或调度容量已变化，请刷新后重试。';
  else if (status >= 500) message = '服务器处理失败，请查看详情或联系管理员。';
  return text ? `${message} 详情：${text}` : `${message}${status ? `（HTTP ${status}）` : ''}`;
}

export async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.body instanceof FormData ? 600000 : 20000);
  try {
    const response = await fetch(`/api${path}`, { ...options, signal: controller.signal });
    let data;
    try { data = await response.json(); } catch { data = null; }
    if (!response.ok) {
      const error = new Error(friendlyError(data?.detail, response.status));
      error.status = response.status;
      throw error;
    }
    if (data === null) throw new Error('服务器未返回有效 JSON，请检查 API 服务。');
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error(options.method === 'POST' ? '服务器响应超时，提交结果尚未确认。请先刷新任务记录，避免重复提交。' : '服务器响应超时，请检查连接后重试。');
    if (error instanceof TypeError) throw new Error(options.method === 'POST' ? '连接中断，操作结果尚未确认。请先刷新任务记录，避免重复提交。' : '无法连接服务器，请检查网络与后端服务。');
    throw error;
  } finally { clearTimeout(timeout); }
}

export const isActive = job => ['queued', 'running'].includes(job?.status);
export const canRetry = job => ['failed', 'cancelled', 'interrupted'].includes(job?.status);
const count = value => Number.isInteger(value) && value >= 0;
export function schedulerState(system, jobs = []) {
  const scheduler = system?.scheduler || {};
  const running = count(scheduler.running_count) ? scheduler.running_count : Math.max(jobs.filter(job => job.status === 'running').length, system?.active_job_ids?.length || (system?.active_job_id ? 1 : 0));
  const queued = count(scheduler.queued_count) ? scheduler.queued_count : jobs.filter(job => job.status === 'queued').length;
  const concurrent = count(scheduler.max_concurrent) && scheduler.max_concurrent > 0 ? scheduler.max_concurrent : 1;
  const maxQueue = count(scheduler.max_queue) ? scheduler.max_queue : null;
  const slots = count(scheduler.available_queue_slots) ? scheduler.available_queue_slots : maxQueue == null ? null : Math.max(0, maxQueue - queued);
  return { running, queued, concurrent, maxQueue, slots, schedulerReady: scheduler.ready !== false, configurationError: backendDetail(scheduler.configuration_error), full: running >= concurrent && slots === 0, willQueue: running >= concurrent || queued > 0, devices: Array.isArray(scheduler.gpu_devices) ? scheduler.gpu_devices : [] };
}
export function submissionState({ system, jobs, busy, connected, ready }) {
  const capacity = schedulerState(system, jobs);
  return { ...capacity, disabled: Boolean(busy || !connected || !ready || !capacity.schedulerReady || capacity.full), label: busy ? '正在提交…' : capacity.willQueue ? '加入队列' : '开始重建' };
}

const stages = { queued: '等待调度', starting: '启动工作进程', preprocessing: '影像预处理', keyframe_selection: '筛选关键帧', chunking: '影像分块', masking: '生成天空掩码', aligning: '对齐与拼接分块', exporting_glb: '导出三维模型', exporting_ply: '导出点云', downsampling: '点云降采样', reconstructing: '三维重建', loading_model: '加载重建模型', loading_models: '加载重建模型', inference: '重建推理', completed: '重建完成', failed: '执行失败', cancelled: '任务已取消', interrupted: '任务已中断' };
export function jobPhase(job) {
  if (!job) return '—';
  if (job.status === 'queued') return Number.isInteger(job.queue_position) && job.queue_position > 0 ? `排队第 ${job.queue_position} 位` : '等待调度（队列位置更新中）';
  if (job.status === 'running' && !job.started_at) return '正在分配执行资源，尚未开始';
  if (!isActive(job)) return stages[job.status] || '状态更新中';
  return stages[job.stage] || (job.stage ? `处理阶段：${job.stage}` : '等待流水线更新阶段');
}
export function shouldPollFast(system, jobs, connected) {
  const capacity = schedulerState(system, jobs);
  return !connected || capacity.running > 0 || capacity.queued > 0 || jobs.some(isActive);
}
export const modelKey = (jobId, output) => `${jobId}:${output.url}`;
export function previewAction({ key, requestedKey, loadingKey, loadedKey, ready }) {
  if (!ready || key !== requestedKey) return 'idle';
  if (key === loadedKey) return 'reuse';
  if (key === loadingKey) return 'wait';
  return 'load';
}

export function formatBytes(bytes) {
  if (bytes == null || bytes === '' || !Number.isFinite(Number(bytes)) || Number(bytes) < 0) return '大小未知';
  const value = Number(bytes);
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
}

export function safeUrl(value) {
  if (typeof value !== 'string' || !value.trim()) return null;
  try {
    const url = new URL(value, window.location.href);
    return ['http:', 'https:'].includes(url.protocol) && url.origin === window.location.origin ? url.href : null;
  } catch { return null; }
}
