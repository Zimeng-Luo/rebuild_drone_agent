import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import * as helpers from '../src/api.js';

const source = await readFile(new URL('../src/main.js', import.meta.url), 'utf8');
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
class Element {
  constructor() { this.hidden = false; this.disabled = false; this.value = ''; this.dataset = {}; this.listeners = {}; this.children = []; this.textContent = ''; this.attributes = {}; this.classList = { add() {}, remove() {}, toggle() {} }; }
  set innerHTML(html) { this.html = html; this.buttons = [...html.matchAll(/data-output-index="(\d+)"/g)].map(match => { const button = new Element(); button.dataset.outputIndex = match[1]; return button; }); }
  get innerHTML() { return this.html || ''; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  querySelector() { return this.span ||= new Element(); }
  querySelectorAll(selector) { return selector === '.view-output' ? this.buttons || [] : []; }
  setAttribute(key, value) { this.attributes[key] = value; }
  append(...nodes) { this.children.push(...nodes); }
  appendChild(node) { this.children.push(node); }
  reportValidity() { return true; }
}
function workbench(api) {
  const nodes = new Map();
  const element = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  const context = {
    ...helpers, active: helpers.isActive, api,
    safeUrl: value => typeof value === 'string' && value.startsWith('/') ? `http://localhost${value}` : null,
    document: { hidden: false, getElementById: element, querySelector: () => element('app'), querySelectorAll: selector => [...selector.matchAll(/#([\w-]+)/g)].map(match => element(match[1])), addEventListener() {}, createElement: () => new Element() },
    localStorage: { getItem: () => null, setItem() {} },
    window: { addEventListener() {} },
    setTimeout: () => 1, clearTimeout() {}, console,
    FormData, AbortController, URL
  };
  const executable = source.split('\n').filter(line => !line.startsWith('import ')).join('\n').replace('updateStart();\ninitializeViewer();\nloadInputs();\nrefresh();', '');
  vm.runInNewContext(`${executable}\nglobalThis.harness = { state, renderSelected, renderJobs, updateStart, selectJob, loadJobs, loadModel, refresh, schedulePoll };`, context);
  const ui = context.harness;
  ui.state.connected = true;
  ui.state.system = { scheduler: { max_concurrent: 1, max_queue: 8, running_count: 1, queued_count: 0, available_queue_slots: 8, gpu_devices: ['0'] } };
  ui.state.source = 'path'; element('input-path').value = '/data/input.mp4';
  return { ...ui, element };
}
const job = (id, status, extra = {}) => ({ id, status, stage: status, created_at: '2026-10-30T00:00:00Z', started_at: status === 'running' ? '2026-10-30T00:00:01Z' : null, input_path: '/input', outputs: [], ...extra });

test('real form remains editable and submits while another job runs; double submit is guarded', async () => {
  const pending = deferred(); let creates = 0;
  const running = job('running', 'running'); const queued = job('new', 'queued', { queue_position: 1 });
  const ui = workbench(async path => {
    if (path === '/jobs') { creates++; return pending.promise; }
    if (path === '/jobs/new') return queued;
    if (path === '/system') return ui.state.system;
    throw new Error(`Unexpected ${path}`);
  });
  ui.state.jobs = [running]; ui.updateStart();
  assert.equal(ui.element('start-button').disabled, false);
  assert.equal(ui.element('task-name').disabled, false);
  assert.equal(ui.element('start-button').querySelector('span').textContent, '加入队列');
  const submit = ui.element('task-form').listeners.submit;
  const first = submit({ preventDefault() {} }); const second = submit({ preventDefault() {} });
  assert.equal(creates, 1);
  pending.resolve(queued); await Promise.all([first, second]);
  assert.equal(ui.state.selected.id, 'new');
  assert.match(ui.element('job-detail').innerHTML, /排队第 1 位.*尚未开始/);
  assert.equal(ui.state.jobs.find(item => item.id === 'running').status, 'running');
});

test('unconfigured scheduler disables create and retry, shows backend detail and leaves fields editable', async () => {
  let calls = 0;
  const ui = workbench(async () => { calls++; throw new Error('Unexpected request'); });
  const failed = job('failed', 'failed');
  ui.state.selected = failed; ui.state.selectedId = failed.id; ui.state.jobs = [failed];
  ui.state.system.scheduler.ready = false;
  ui.state.system.scheduler.configuration_error = 'GPU_DEVICES missing';
  ui.renderSelected();
  assert.equal(ui.element('start-button').disabled, true);
  assert.equal(ui.element('retry-job').disabled, true);
  assert.equal(ui.element('task-name').disabled, false);
  assert.match(ui.element('queue-capacity').textContent, /未配置任务设备.*GPU_DEVICES missing/);
  await ui.element('task-form').listeners.submit({ preventDefault() {} });
  await ui.element('retry-job').listeners.click();
  assert.equal(calls, 0);
  ui.state.system.scheduler.ready = true; ui.updateStart();
  assert.equal(ui.element('start-button').disabled, false);
  assert.equal(ui.element('retry-job').disabled, false);
});

test('real retry creates once, preserves original record and does not switch unrelated selection', async () => {
  const pending = deferred(); let retries = 0;
  const old = job('old', 'failed'); const other = job('other', 'running');
  const next = job('retry', 'queued', { queue_position: 1, retry_of: 'old' });
  const ui = workbench(async path => {
    if (path === '/jobs/old/retry') { retries++; return pending.promise; }
    if (path === '/system') return ui.state.system;
    throw new Error(`Unexpected ${path}`);
  });
  ui.state.jobs = [old, other]; ui.state.selected = old; ui.state.selectedId = old.id; ui.renderSelected();
  const retry = ui.element('retry-job').listeners.click;
  const first = retry(); const second = retry(); assert.equal(retries, 1);
  ui.state.selected = other; ui.state.selectedId = other.id; ui.renderSelected();
  pending.resolve(next); await Promise.all([first, second]);
  assert.equal(ui.state.selected.id, 'other');
  assert.equal(ui.state.jobs.find(item => item.id === 'old').status, 'failed');
  assert.equal(ui.state.jobs.find(item => item.id === 'retry').retry_of, 'old');
  ui.state.selected = old; ui.state.selectedId = old.id; ui.renderSelected();
  assert.equal(ui.element('retry-job').disabled, true);
  await retry(); assert.equal(retries, 1);
});

test('queued cancellation remains bound to the original job while the user selects another', async () => {
  const pending = deferred(); const waiting = job('wait', 'queued', { queue_position: 2 }); const running = job('run', 'running');
  const ui = workbench(async path => path === '/system' ? ui.state.system : pending.promise);
  ui.state.jobs = [waiting, running]; ui.state.selected = waiting; ui.state.selectedId = waiting.id; ui.renderSelected();
  assert.match(ui.element('cancel-job').innerHTML, /取消排队/);
  const cancel = ui.element('cancel-job').listeners.click();
  ui.state.selected = running; ui.state.selectedId = running.id; ui.renderSelected();
  assert.equal(ui.element('cancel-job').disabled, false);
  pending.resolve({ ...waiting, status: 'cancelled' }); await cancel;
  assert.equal(ui.state.selected.id, 'run'); assert.equal(ui.state.selected.status, 'running');
  assert.equal(ui.state.jobs.find(item => item.id === 'wait').status, 'cancelled');
});

test('stale in-flight list response cannot overwrite a newly created job', async () => {
  const list = deferred(); const old = job('old', 'running'); const next = job('new', 'queued', { queue_position: 1 });
  const ui = workbench(async (path, options) => {
    if (path === '/jobs' && options?.method === 'POST') return next;
    if (path === '/jobs') return list.promise;
    if (path === '/jobs/new') return next;
    if (path === '/system') return ui.state.system;
    throw new Error(`Unexpected ${path}`);
  });
  ui.state.jobs = [old]; ui.updateStart();
  const poll = ui.loadJobs();
  await ui.element('task-form').listeners.submit({ preventDefault() {} });
  list.resolve({ jobs: [old] }); await poll;
  assert.equal(ui.state.selected.id, 'new');
  assert.ok(ui.state.jobs.some(item => item.id === 'new'));
});

test('stale in-flight detail cannot restore queued state after cancellation', async () => {
  const detail = deferred(); const waiting = job('wait', 'queued', { queue_position: 1 });
  const ui = workbench(async path => {
    if (path === '/jobs/wait') return detail.promise;
    if (path === '/jobs/wait/cancel') return { ...waiting, status: 'cancelled' };
    if (path === '/system') return ui.state.system;
    throw new Error(`Unexpected ${path}`);
  });
  ui.state.jobs = [waiting];
  const selecting = ui.selectJob('wait');
  await ui.element('cancel-job').listeners.click();
  detail.resolve(waiting); await selecting;
  assert.equal(ui.state.selected.status, 'cancelled');
});

test('history, refresh and selection never load a model; explicit clicks coalesce and reuse the loaded model', async () => {
  const pending = deferred(); let loads = 0;
  const result = job('result', 'completed', { outputs: [{ name: 'scene.glb', url: '/model.glb', size_bytes: 1048576 }] });
  const other = job('other', 'running');
  const ui = workbench(async path => path === '/jobs/result' ? result : other);
  ui.state.viewer = { load: () => { loads++; return pending.promise; }, resize() {} };
  ui.state.jobs = [result, other]; ui.state.selected = result; ui.state.selectedId = result.id;
  ui.renderSelected(); ui.renderSelected();
  assert.equal(loads, 0); assert.match(ui.element('empty-message').textContent, /1.0 MB/);
  const view = ui.element('outputs').querySelectorAll('.view-output')[0];
  view.listeners.click(); view.listeners.click(); assert.equal(loads, 1);
  pending.resolve(true); await new Promise(resolve => setImmediate(resolve));
  ui.renderSelected(); view.listeners.click(); assert.equal(loads, 1);
  await ui.selectJob('other'); await ui.selectJob('result');
  assert.equal(loads, 1); assert.equal(ui.element('canvas-host').hidden, true);
  ui.element('outputs').querySelectorAll('.view-output')[0].listeners.click();
  assert.equal(loads, 1); assert.equal(ui.element('canvas-host').hidden, false);
});
