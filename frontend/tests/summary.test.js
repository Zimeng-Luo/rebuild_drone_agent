import test from 'node:test';
import assert from 'node:assert/strict';
import { summaryTotal, summaryPhases, summaryBreakdown, summaryFrames, summaryChunks, summarySpace, summaryPeaks, summaryLoop, summaryAvailable } from '../src/summary.js';

// Shape mirrors a real task_summary.log produced by the pipeline (see the
// completed 主楼02 run): aggregate rows plus per-segment leaf steps.
const summary = {
  basic: { task_created_at: '2026-11-14 21:38:43', summary_created_at: '2026-11-14 21:46:29' },
  time: {
    total: { name: 'total', duration_seconds: 466.008 },
    steps: [
      { name: 'read_video_segment_0', duration_seconds: 4.1 },
      { name: 'stream_scan_segment_0', duration_seconds: 5.3 },
      { name: 'key_frame_select_global', duration_seconds: 12.912 },
      { name: 'data_preprocess', duration_seconds: 141.829 },
      { name: 'rebuild_model_load', duration_seconds: 10.64 },
      { name: 'rebuild_chunk_0', duration_seconds: 3.417 },
      { name: 'rebuild_chunks', duration_seconds: 26.173 },
      { name: 'align_compute_and_save', duration_seconds: 69.522 },
      { name: 'align_export_glb', duration_seconds: 113.709 },
      { name: 'align_export_ply', duration_seconds: 114.369 },
      { name: 'align_and_merge', duration_seconds: 297.602 }
    ]
  },
  space: {
    total_mb: 1086.8,
    by_suffix: { '.npz': { mb: 704.743 }, '.glb': { mb: 197.179 }, '.ply': { mb: 184.854 }, '.log': { mb: 0.022 } }
  },
  parameters: {
    chunk_count: 5,
    run_evidence: {
      preprocess: { sampled_frame_count: 387, selected_frame_count: 81, selection_retention_ratio: 0.20930232558139536, chunk_count: 5, per_chunk_frame_counts: [30, 30, 30, 30, 21] },
      alignment: { mode: 'loop_optimized', loop_candidate_count: 3, loop_accepted_count: 3, independent_loop_verified: false }
    }
  },
  performance: { total: { cpu_percent: { mean: 834.29, peak: 14230.501 }, memory_rss_mb: { mean: 4483.456, peak: 9678.094 }, gpu_memory_allocated_mb: { mean: 462.356, peak: 8491.449 } } }
};

test('total duration comes from the pipeline total row, not a sum of steps', () => {
  assert.equal(summaryTotal(summary), 466.008);
  assert.equal(summaryTotal({}), null);
  assert.equal(summaryTotal(null), null);
});

test('phase shares use the pipeline aggregate rows so nested steps are never double counted', () => {
  const phases = summaryPhases(summary);
  assert.deepEqual(phases.map(phase => phase.key), ['data_preprocess', 'rebuild_chunks', 'align_and_merge']);
  assert.deepEqual(phases.map(phase => phase.seconds), [141.829, 26.173, 297.602]);
  assert.ok(Math.abs(phases[2].share - 297.602 / 466.008) < 1e-9);
  const covered = phases.reduce((sum, phase) => sum + phase.seconds, 0);
  assert.ok(covered <= 466.008 + 1e-6);
});

test('phase aggregation falls back to leaf steps when no aggregate row was written', () => {
  const leafOnly = { time: { total: { duration_seconds: 10 }, steps: [
    { name: 'read_video_segment_0', duration_seconds: 2 },
    { name: 'stream_scan_segment_0', duration_seconds: 2 },
    { name: 'rebuild_chunk_0', duration_seconds: 3 },
    { name: 'align_export_ply', duration_seconds: 2.5 }
  ] } };
  const phases = summaryPhases(leafOnly);
  assert.deepEqual(phases.map(phase => [phase.key, phase.seconds]), [
    ['data_preprocess', 4], ['rebuild_chunks', 3], ['align_and_merge', 2.5]
  ]);
});

test('breakdown ranks leaf steps by cost and excludes the aggregate rows', () => {
  const rows = summaryBreakdown(summary, 3);
  assert.deepEqual(rows.map(row => row.name), ['align_export_ply', 'align_export_glb', 'align_compute_and_save']);
  assert.ok(rows.every(row => !['data_preprocess', 'rebuild_chunks', 'align_and_merge'].includes(row.name)));
  assert.ok(Math.abs(rows[0].share - 114.369 / 466.008) < 1e-9);
  // The nested per-segment steps stay out of the headline list.
  assert.ok(!summaryBreakdown(summary, 20).some(row => row.name.startsWith('read_video_segment')));
});

test('frames, chunks, space and peaks read the nested evidence without throwing', () => {
  const frames = summaryFrames(summary);
  assert.equal(frames.sampled, 387);
  assert.equal(frames.selected, 81);
  assert.ok(Math.abs(frames.retention - 0.2093023) < 1e-6);
  assert.deepEqual(summaryChunks(summary), { count: 5, perChunk: [30, 30, 30, 30, 21] });
  const space = summarySpace(summary);
  assert.equal(space.totalMb, 1086.8);
  // Sub-megabyte suffixes (.log/.sum/.json) stay out of the digest row.
  assert.deepEqual(space.groups.map(group => group.suffix), ['.npz', '.glb', '.ply']);
  assert.ok(Math.abs(space.groups[0].share - 704.743 / 1086.8) < 1e-9);
  assert.deepEqual(summaryPeaks(summary), { gpuMb: 8491.449, rssMb: 9678.094, cpuPercent: 14230.501 });
  assert.deepEqual(summaryLoop(summary), { mode: 'loop_optimized', accepted: 3, candidates: 3, verified: false });
});

test('every reader degrades to null or empty on a partial, foreign or empty summary', () => {
  for (const value of [null, undefined, {}, { time: null }, { time: { steps: 'nope' } }, { space: { by_suffix: { '.glb': {} } } }]) {
    assert.doesNotThrow(() => {
      summaryTotal(value); summaryPhases(value); summaryBreakdown(value); summaryFrames(value);
      summaryChunks(value); summarySpace(value); summaryPeaks(value); summaryLoop(value); summaryAvailable(value);
    });
  }
  assert.deepEqual(summaryPhases({}), []);
  assert.deepEqual(summaryBreakdown({}), []);
  assert.equal(summaryFrames({}), null);
  assert.equal(summarySpace({ space: { by_suffix: { '.glb': {} } } }), null);
  assert.equal(summaryPeaks({ performance: { total: {} } }), null);
  // Steps with unusable durations must not become NaN rows.
  assert.deepEqual(summaryBreakdown({ time: { total: { duration_seconds: 5 }, steps: [{ name: 'x', duration_seconds: 'abc' }, { name: '', duration_seconds: 1 }] } }), []);
});

test('a list entry without the diagnostics payload still advertises that a digest exists', () => {
  assert.equal(summaryAvailable({ summary: null, has_summary: true }), true);
  assert.equal(summaryAvailable({ summary: { time: {} } }), true);
  assert.equal(summaryAvailable({ summary: null, has_summary: false }), false);
  assert.equal(summaryAvailable({}), false);
});
