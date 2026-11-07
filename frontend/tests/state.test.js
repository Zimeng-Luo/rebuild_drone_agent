import test from 'node:test';
import assert from 'node:assert/strict';
import { api, backendDetail, friendlyError, schedulerState, submissionState, jobPhase, shouldPollFast, canRetry, previewAction, modelKey, formatBytes } from '../src/api.js';

const system = (running = 1, queued = 0, maxQueue = 8, concurrent = 1) => ({ scheduler: { running_count: running, queued_count: queued, max_concurrent: concurrent, max_queue: maxQueue, available_queue_slots: Math.max(0, maxQueue - queued), gpu_devices: ['0', '1'] } });

test('running jobs permit queue submission; capacity and connection control disabling', () => {
  const base = { system: system(), jobs: [{ status: 'running' }], connected: true, ready: true, busy: false };
  assert.equal(submissionState(base).disabled, false);
  assert.equal(submissionState(base).label, '加入队列');
  for (const patch of [{ busy: true }, { connected: false }, { ready: false }, { system: system(1, 8) }]) assert.equal(submissionState({ ...base, ...patch }).disabled, true);
  assert.equal(submissionState({ ...base, system: system(0) }).label, '开始重建');
  assert.equal(submissionState({ ...base, system: system(0, 0, 0) }).disabled, false);
  assert.equal(submissionState({ ...base, system: system(1, 0, 0) }).disabled, true);
  assert.equal(submissionState({ ...base, system: { ...system(), gpu: { available: false } } }).disabled, false);
  assert.equal(submissionState({ ...base, system: system(1, 0, 8, 2) }).label, '开始重建');
});

test('scheduler readiness disables submission and preserves configuration details independently of GPU discovery', () => {
  const unconfigured = { ...system(0), gpu: { available: true } };
  unconfigured.scheduler.ready = false;
  unconfigured.scheduler.configuration_error = 'GPU_DEVICES must be configured explicitly';
  const base = { system: unconfigured, jobs: [], connected: true, ready: true, busy: false };
  assert.equal(submissionState(base).disabled, true);
  assert.equal(submissionState(base).schedulerReady, false);
  assert.equal(submissionState(base).configurationError, 'GPU_DEVICES must be configured explicitly');
  unconfigured.scheduler.ready = true;
  unconfigured.gpu.available = false;
  assert.equal(submissionState(base).disabled, false);
});

test('scheduler counts are authoritative; legacy active ID still keeps active capacity visible', () => {
  assert.equal(schedulerState(system(0), [{ status: 'running' }]).running, 0);
  assert.equal(schedulerState({ active_job_ids: ['a', 'b'] }).running, 2);
  assert.equal(schedulerState({ active_job_id: 'a' }).running, 1);
  assert.equal(schedulerState(null, [{ status: 'queued' }]).queued, 1);
  assert.equal(shouldPollFast(system(1, 2), [], true), true);
  assert.equal(shouldPollFast(system(0), [], true), false);
});

test('queued and missing-start jobs never claim running stages', () => {
  assert.equal(jobPhase({ status: 'queued', queue_position: 2, stage: 'inference', started_at: null }), '排队第 2 位');
  assert.match(jobPhase({ status: 'queued', queue_position: null }), /位置更新中/);
  assert.match(jobPhase({ status: 'running', stage: 'inference', started_at: null }), /尚未开始/);
  assert.equal(jobPhase({ status: 'running', stage: 'exporting_glb', started_at: '2026-10-30' }), '导出三维模型');
  assert.match(jobPhase({ status: 'running', stage: 'custom_phase', started_at: '2026-10-30' }), /custom_phase/);
  for (const status of ['failed', 'cancelled', 'interrupted']) assert.equal(canRetry({ status }), true);
  for (const status of ['queued', 'running', 'completed']) assert.equal(canRetry({ status }), false);
});

test('preview requires explicit matching request, coalesces pending loads, reuses completed loads', () => {
  const key = modelKey('a', { url: '/model.glb' });
  const args = { key, ready: true, loadingKey: null, loadedKey: null, requestedKey: null };
  assert.equal(previewAction(args), 'idle');
  assert.equal(previewAction({ ...args, requestedKey: key }), 'load');
  assert.equal(previewAction({ ...args, requestedKey: key, loadingKey: key }), 'wait');
  assert.equal(previewAction({ ...args, requestedKey: key, loadingKey: key, loadedKey: key }), 'reuse');
  assert.equal(previewAction({ ...args, requestedKey: key, loadedKey: key, ready: false }), 'idle');
});

test('friendly errors preserve all backend details and report unknown sizes honestly', () => {
  assert.match(friendlyError('Queue is full (8 waiting)', 409), /队列已满.*Queue is full \(8 waiting\)/);
  assert.match(friendlyError('GPU devices must be configured explicitly', 503), /GPU.*配置.*详情/);
  assert.equal(backendDetail([{ loc: ['body', 'input_path'], msg: 'missing', type: 'required' }]), 'body.input_path: missing');
  assert.match(friendlyError({ reason: 'disk full' }, 500), /disk full/);
  assert.equal(formatBytes(null), '大小未知');
  assert.equal(formatBytes(undefined), '大小未知');
  assert.equal(formatBytes(0), '0 B');
  assert.equal(formatBytes(1024), '1.0 KB');
});

test('API errors expose HTTP status and translated backend details', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false, status: 409, json: async () => ({ detail: 'Queue is full' }) });
  try { await assert.rejects(api('/jobs', { method: 'POST' }), error => error.status === 409 && /队列已满.*Queue is full/.test(error.message)); }
  finally { globalThis.fetch = original; }
});
