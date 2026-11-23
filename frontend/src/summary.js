// Structured, defensive view over the pipeline's task_summary.log payload.
// The pipeline writes a large diagnostic document (per-step timings, cleanup
// lists, full parameter dump). The workbench only needs a small, stable digest
// of it, so every reader here tolerates missing or oddly shaped fields.

const AGGREGATES = [
  ['data_preprocess', '影像预处理', '读取 · 筛选 · 天空掩码'],
  ['rebuild_chunks', '分块重建', 'VGGT 逐块推理'],
  ['align_and_merge', '对齐与拼接', '配准 · 导出模型']
];

const PREPROCESS_PREFIXES = ['read_video_segment', 'stream_scan_segment', 'key_frame_select', 'reload_selected', 'chunk_split', 'chunk_save', 'remove_mask'];
const REBUILD_PREFIXES = ['rebuild_chunk', 'rebuild_model_load'];
const ALIGN_PREFIXES = ['align_compute_and_save', 'align_export_glb', 'align_export_ply'];

const num = value => (typeof value === 'number' && Number.isFinite(value) ? value : null);
const seconds = value => {
  const direct = num(value);
  if (direct != null) return direct;
  const parsed = Number(value);
  return Number.isFinite(parsed) && value !== null && value !== '' ? parsed : null;
};

function stepsOf(summary) {
  const steps = summary?.time?.steps;
  if (!Array.isArray(steps)) return [];
  return steps
    .map(step => ({ name: String(step?.name ?? ''), seconds: seconds(step?.duration_seconds) }))
    .filter(step => step.name && step.seconds != null && step.seconds >= 0);
}

function bucket(name) {
  if (name.startsWith('rebuild')) return 'rebuild_chunks';
  if (name.startsWith('align')) return 'align_and_merge';
  if (name === 'data_preprocess') return 'data_preprocess';
  if (PREPROCESS_PREFIXES.some(prefix => name.startsWith(prefix))) return 'data_preprocess';
  if (REBUILD_PREFIXES.some(prefix => name.startsWith(prefix))) return 'rebuild_chunks';
  if (ALIGN_PREFIXES.some(prefix => name.startsWith(prefix))) return 'align_and_merge';
  return null;
}

export function summaryTotal(summary) {
  return seconds(summary?.time?.total?.duration_seconds) ?? seconds(summary?.time?.total);
}

export function summaryPhases(summary) {
  const total = summaryTotal(summary);
  const steps = stepsOf(summary);
  const sums = new Map();
  for (const step of steps) {
    const key = bucket(step.name);
    if (!key) continue;
    sums.set(key, (sums.get(key) || 0) + step.seconds);
  }
  // Prefer the pipeline's own aggregate rows; they include the per-segment
  // bookkeeping that a naive sum of leaf steps would miss.
  for (const [key] of AGGREGATES) {
    const aggregate = steps.find(step => step.name === key);
    if (aggregate) sums.set(key, aggregate.seconds);
  }
  return AGGREGATES
    .map(([key, label, hint]) => {
      const value = sums.get(key);
      if (value == null || value <= 0) return null;
      return { key, label, hint, seconds: value, share: total && total > 0 ? value / total : null };
    })
    .filter(Boolean);
}

const perSegment = /^(read_video_segment|stream_scan_segment)_\d+$/;

export function summaryBreakdown(summary, limit = 8) {
  const total = summaryTotal(summary);
  return stepsOf(summary)
    .filter(step => !AGGREGATES.some(([key]) => key === step.name))
    .filter(step => !perSegment.test(step.name))
    .sort((a, b) => b.seconds - a.seconds)
    .slice(0, Math.max(1, limit))
    .map(step => ({ ...step, share: total && total > 0 ? step.seconds / total : null }));
}

export function summaryFrames(summary) {
  const preprocess = summary?.parameters?.run_evidence?.preprocess || summary?.preprocess || {};
  const sampled = num(preprocess.sampled_frame_count);
  const selected = num(preprocess.selected_frame_count);
  const retention = num(preprocess.selection_retention_ratio);
  if (sampled == null && selected == null) return null;
  return {
    sampled,
    selected,
    retention: retention ?? (sampled && selected != null ? selected / sampled : null),
    basis: typeof preprocess.frame_count_basis === 'string' ? preprocess.frame_count_basis : ''
  };
}

export function summaryChunks(summary) {
  const preprocess = summary?.parameters?.run_evidence?.preprocess || summary?.preprocess || {};
  const chunks = num(preprocess.chunk_count) ?? num(summary?.parameters?.chunk_count);
  const counts = Array.isArray(preprocess.per_chunk_frame_counts) ? preprocess.per_chunk_frame_counts.filter(value => num(value) != null) : [];
  return { count: chunks, perChunk: counts };
}

export function summarySpace(summary) {
  const space = summary?.space;
  const totalMb = num(space?.total_mb) ?? (num(space?.total_bytes) != null ? space.total_bytes / (1024 ** 2) : null);
  const bySuffix = space?.by_suffix && typeof space.by_suffix === 'object' ? space.by_suffix : {};
  const groups = Object.entries(bySuffix)
    .map(([suffix, value]) => ({ suffix, mb: num(value?.mb) ?? (num(value?.bytes) != null ? value.bytes / (1024 ** 2) : null) }))
    .filter(group => group.mb != null && group.mb > 0)
    .sort((a, b) => b.mb - a.mb);
  if (totalMb == null && !groups.length) return null;
  // Suffixes below a tenth of a megabyte only add noise to the digest row.
  const shown = groups.filter(group => group.mb >= 0.1).slice(0, 5);
  return { totalMb, groups: shown.map(group => ({ ...group, share: totalMb && totalMb > 0 ? group.mb / totalMb : null })) };
}

export function summaryPeaks(summary) {
  const total = summary?.performance?.total;
  if (!total || typeof total !== 'object') return null;
  const gpu = num(total.gpu_memory_allocated_mb?.peak);
  const rss = num(total.memory_rss_mb?.peak);
  const cpu = num(total.cpu_percent?.peak);
  if (gpu == null && rss == null && cpu == null) return null;
  return { gpuMb: gpu, rssMb: rss, cpuPercent: cpu };
}

export function summaryLoop(summary) {
  const evidence = summary?.parameters?.run_evidence;
  const alignment = evidence?.alignment || {};
  const mode = typeof alignment.mode === 'string' ? alignment.mode : null;
  const accepted = num(alignment.loop_accepted_count);
  const candidates = num(alignment.loop_candidate_count);
  if (!mode && accepted == null) return null;
  return { mode, accepted, candidates, verified: alignment.independent_loop_verified === true };
}

// An explicitly shallow digest used when a call site only needs to know whether
// a full summary exists (the job list endpoint no longer ships the payload).
export function summaryAvailable(job) {
  if (job?.has_summary === true) return true;
  const summary = job?.summary;
  return Boolean(summary && typeof summary === 'object' && Object.keys(summary).length);
}
