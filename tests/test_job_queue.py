"""FIFO, GPU isolation and lifecycle tests with small real CPU subprocesses."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.jobs import BusyError, JobManager, atomic_json
from app.server import create_app

UUID_A = 'GPU-11111111-2222-3333-4444-555555555555'
UUID_B = 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
CPU_WORKER = r'''
import json, os, pathlib, signal, subprocess, sys, time
folder = pathlib.Path(sys.argv[1])
job = json.loads((folder / 'job.json').read_text())
name = job['name']
child = None
if name.startswith('tree'):
    child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(90)'])
started = {'pid':os.getpid(), 'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'), 'child':child.pid if child else None}
(folder / 'started.tmp').write_text(json.dumps(started))
(folder / 'started.tmp').replace(folder / 'started.json')
print('[workbench-stage] rebuilding', flush=True)
if name.startswith(('hold', 'tree', 'fail')):
    while not (folder / 'release').exists():
        time.sleep(0.01)
if name.startswith('fail'):
    (folder / 'result.json').write_text(json.dumps({'status':'failed','error':'CPU fixture failure'}))
    sys.exit(7)
(folder / 'result.json').write_text(json.dumps({'status':'completed','summary':started}))
'''


def command(folder):
    job = json.loads((folder / 'job.json').read_text())
    if job['name'] == 'spawn_error':
        return ['/does/not/exist/fixture-worker']
    return [sys.executable, '-u', '-c', CPU_WORKER, str(folder)]


def until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    pytest.fail('CPU fixture did not reach the expected state')


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def folder(manager, job):
    return manager.jobs_dir / job['id']


def started(manager, job):
    marker = folder(manager, job) / 'started.json'
    until(marker.exists)
    return json.loads(marker.read_text())


def release(manager, job):
    (folder(manager, job) / 'release').touch()


def finished(manager, job, status='completed'):
    until(lambda: manager.get(job['id'])['status'] == status)
    return manager.get(job['id'])


@contextmanager
def scheduler(root, **kwargs):
    manager = JobManager(root, command, **kwargs)
    manager.start()
    try:
        yield manager
    finally:
        manager.close()


def enqueue(manager, name):
    return manager.submit('fixture.mp4', 'video', name, {})


def test_fifo_queue_positions_and_no_queued_process(tmp_path):
    with scheduler(tmp_path) as manager:
        first = enqueue(manager, 'hold-first')
        started(manager, first)
        second = enqueue(manager, 'hold-second')
        third = enqueue(manager, 'hold-third')
        assert second['queue_position'] == 1 and third['queue_position'] == 2
        assert second['gpu_device'] is None and third['started_at'] is None
        assert not (folder(manager, second) / 'process.json').exists()
        assert not (folder(manager, third) / 'stdout.log').exists()
        release(manager, first)
        started(manager, second)
        assert manager.get(third['id'])['queue_position'] == 1
        assert not (folder(manager, third) / 'started.json').exists()
        release(manager, second)
        started(manager, third)
        release(manager, third)
        jobs = [finished(manager, item) for item in (first, second, third)]
        assert all(job['queue_position'] is None for job in jobs)
        assert jobs[0]['finished_at'] <= jobs[1]['started_at'] <= jobs[1]['finished_at'] <= jobs[2]['started_at']


def test_saturation_and_queued_cancel_do_not_touch_running(tmp_path):
    with scheduler(tmp_path, max_queue=2) as manager:
        first = enqueue(manager, 'hold-first')
        pid = started(manager, first)['pid']
        second = enqueue(manager, 'hold-second')
        third = enqueue(manager, 'hold-third')
        with pytest.raises(BusyError, match='queue is full'):
            enqueue(manager, 'overflow')
        state = manager.scheduler_state()
        assert state['active_job_ids'] == [first['id']]
        assert state['scheduler']['queued_count'] == 2
        assert state['scheduler']['available_queue_slots'] == 0
        assert manager.cancel(second['id'])['status'] == 'cancelled'
        assert alive(pid) and manager.get(first['id'])['status'] == 'running'
        assert manager.get(third['id'])['queue_position'] == 1
        replacement = enqueue(manager, 'hold-replacement')
        assert replacement['queue_position'] == 2
        assert not (folder(manager, second) / 'process.json').exists()


def test_two_slots_bind_env_and_cancel_only_target_group(tmp_path):
    with scheduler(tmp_path, max_concurrent=2, gpu_devices='3,7') as manager:
        first = enqueue(manager, 'tree-first')
        second = enqueue(manager, 'tree-second')
        one, two = started(manager, first), started(manager, second)
        assert one['gpu'] == first['gpu_device'] == '3'
        assert two['gpu'] == second['gpu_device'] == '7'
        third = enqueue(manager, 'hold-third')
        assert third['status'] == 'queued'
        assert manager.cancel(first['id'])['status'] == 'cancelled'
        assert not alive(one['pid']) and not alive(one['child'])
        assert alive(two['pid']) and alive(two['child'])
        assert manager.get(second['id'])['status'] == 'running'
        assert started(manager, third)['gpu'] == '3'
        assert manager.scheduler_state()['active_job_ids'] == [second['id'], third['id']]
        manager.cancel(third['id'])
        assert alive(two['pid'])
        manager.cancel(second['id'])
        assert not alive(two['child'])


def test_failures_and_spawn_errors_advance_queue_without_retry(tmp_path):
    with scheduler(tmp_path) as manager:
        first = enqueue(manager, 'fail-first')
        started(manager, first)
        broken = enqueue(manager, 'spawn_error')
        following = enqueue(manager, 'success')
        release(manager, first)
        assert 'CPU fixture failure' in finished(manager, first, 'failed')['error']
        assert 'Could not start worker' in finished(manager, broken, 'failed')['error']
        assert finished(manager, following)['summary']['pid']
        assert len(manager.list()) == 3


def test_shutdown_stops_all_and_never_starts_queued(tmp_path):
    manager = JobManager(tmp_path, command, max_concurrent=2)
    manager.start()
    first, second = enqueue(manager, 'tree-a'), enqueue(manager, 'tree-b')
    processes = [started(manager, job) for job in (first, second)]
    queued = enqueue(manager, 'hold-queued')
    manager.close()
    assert all(manager.get(job['id'])['status'] == 'interrupted' for job in (first, second, queued))
    assert not (folder(manager, queued) / 'started.json').exists()
    assert not (folder(manager, queued) / 'process.json').exists()
    assert all(not alive(pid) for process in processes for pid in (process['pid'], process['child']))
    with pytest.raises(BusyError, match='shutting down'):
        enqueue(manager, 'late')
    with scheduler(tmp_path) as restarted:
        assert {job['status'] for job in restarted.list()} == {'interrupted'}
        assert restarted.scheduler_state()['active_job_ids'] == []


def test_concurrent_submit_never_exceeds_capacity(tmp_path):
    with scheduler(tmp_path, max_concurrent=2, max_queue=3) as manager:
        barrier = threading.Barrier(16)
        def attempt(index):
            barrier.wait()
            try:
                return enqueue(manager, f'hold-{index}')
            except BusyError:
                return None
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(attempt, range(16)))
        assert len([item for item in results if item]) == 5
        state = manager.scheduler_state()['scheduler']
        assert state['running_count'] == 2 and state['queued_count'] == 3
        assert state['available_queue_slots'] == 0


def test_slot_is_reserved_until_process_group_cleanup(tmp_path, monkeypatch):
    import app.jobs as jobs_module
    original_reap = jobs_module.reap_group
    cleaning = threading.Event()
    allow_cleanup = threading.Event()
    target = []
    def delayed_cleanup(pid):
        if target and pid == target[0]:
            cleaning.set()
            allow_cleanup.wait(timeout=5)
        original_reap(pid)
    monkeypatch.setattr(jobs_module, 'reap_group', delayed_cleanup)
    with scheduler(tmp_path) as manager:
        first = enqueue(manager, 'tree-first')
        target.append(started(manager, first)['pid'])
        second = enqueue(manager, 'hold-second')
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                cancellation = pool.submit(manager.cancel, first['id'])
                try:
                    assert cleaning.wait(timeout=3)
                    state = manager.scheduler_state()
                    assert state['active_job_ids'] == [first['id']]
                    assert state['scheduler']['running_count'] == 1
                    assert manager.get(second['id'])['status'] == 'queued'
                    assert not (folder(manager, second) / 'started.json').exists()
                    assert not cancellation.done()
                finally:
                    allow_cleanup.set()
                assert cancellation.result(timeout=5)['status'] == 'cancelled'
            started(manager, second)
        finally:
            allow_cleanup.set()


def test_get_snapshots_do_not_wait_for_artifact_io(tmp_path, monkeypatch):
    scanning = threading.Event()
    proceed = threading.Event()
    with scheduler(tmp_path) as manager:
        original_outputs = manager._outputs
        def slow_outputs(job_id):
            scanning.set()
            proceed.wait(timeout=3)
            return original_outputs(job_id)
        monkeypatch.setattr(manager, '_outputs', slow_outputs)
        first = enqueue(manager, 'hold-first')
        started(manager, first)
        release(manager, first)
        try:
            assert scanning.wait(timeout=3)
            start = time.monotonic()
            assert manager.get(first['id'])['id'] == first['id']
            assert len(manager.list()) == 1
            assert manager.scheduler_state()['scheduler']['running_count'] == 1
            assert time.monotonic() - start < 0.5
        finally:
            proceed.set()
        finished(manager, first)


@pytest.mark.parametrize('kwargs', [
    {'max_concurrent':0}, {'max_concurrent':-1}, {'max_concurrent':True},
    {'max_queue':-1}, {'max_queue':True}, {'max_queue':1.5},
    {'gpu_devices':''}, {'gpu_devices':[]}, {'gpu_devices':'0,'},
    {'gpu_devices':'0,0'}, {'gpu_devices':'01'}, {'gpu_devices':'-1'},
    {'gpu_devices':'GPU-short'}, {'gpu_devices':'0;echo bad'},
    {'gpu_devices':['0', UUID_A]}, {'gpu_devices':[UUID_B, UUID_B.upper().replace('GPU-', 'GPU-')]},
    {'max_concurrent':2, 'gpu_devices':'0'},
])
def test_invalid_scheduler_config_is_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        JobManager(tmp_path, command, **kwargs)


def test_production_parallelism_requires_explicit_gpu_pool(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    with pytest.raises(ValueError, match='explicit'):
        JobManager(tmp_path, max_concurrent=2)
    manager = JobManager(tmp_path, max_concurrent=2, gpu_devices='0,1')
    assert manager.scheduler_state()['scheduler']['ready'] is True


@pytest.mark.parametrize('visible', [None, '', '-1', '0,1', 'invalid'])
def test_unconfigured_gpu_still_allows_api_but_rejects_jobs(tmp_path, monkeypatch, visible):
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    (tmp_path / 'input.mp4').write_bytes(b'cpu-fixture')
    with TestClient(create_app(tmp_path)) as client:
        state = client.get('/api/system').json()['scheduler']
        assert state['ready'] is False and 'nvidia-smi' in state['configuration_error']
        response = client.post('/api/jobs', json=dict(input_path='input.mp4', read_type='video'))
        assert response.status_code == 409
        assert client.get('/api/jobs').json()['jobs'] == []


def test_single_inherited_device_and_explicit_uuid_env(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    inherited = JobManager(tmp_path)
    assert inherited.gpu_devices == ['7'] and inherited.configuration_error is None
    # Replace only the command so the production inheritance branch is tested on CPU.
    inherited.command_factory = command
    inherited.start()
    try:
        job = enqueue(inherited, 'success')
        assert finished(inherited, job)['summary']['gpu'] == '7'
    finally:
        inherited.close()
    with scheduler(tmp_path / 'uuid', max_concurrent=2, gpu_devices=f'{UUID_A},{UUID_B}') as manager:
        one, two = enqueue(manager, 'hold-one'), enqueue(manager, 'hold-two')
        assert started(manager, one)['gpu'] == UUID_A
        assert started(manager, two)['gpu'] == UUID_B
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '7'


def test_zero_queue_accepts_only_immediate_execution(tmp_path):
    with scheduler(tmp_path, max_queue=0) as manager:
        job = enqueue(manager, 'hold-first')
        started(manager, job)
        with pytest.raises(BusyError, match='queue is full'):
            enqueue(manager, 'overflow')
        manager.cancel(job['id'])
        assert enqueue(manager, 'success')['status'] == 'running'


def api_submit(client, name='hold-one', **options):
    response = client.post('/api/jobs', json=dict(input_path='input.mp4', read_type='video', name=name, **options))
    assert response.status_code == 201, response.text
    return response.json()


def test_retry_failed_cancelled_and_interrupted_preserves_original(tmp_path):
    (tmp_path / 'input.mp4').write_bytes(b'fixture')
    app = create_app(tmp_path, command)
    with TestClient(app) as client:
        manager = app.state.manager
        failed = api_submit(client, 'fail-original', image_sample_interval=7, remove_mask_enable=True)
        started(manager, failed)
        release(manager, failed)
        original = finished(manager, failed, 'failed')
        response = client.post(f"/api/jobs/{failed['id']}/retry")
        assert response.status_code == 201
        retry = response.json()
        assert retry['id'] != failed['id'] and retry['retry_of'] == failed['id']
        assert retry['options'] == original['options'] and retry['input_path'] == original['input_path']
        assert manager.get(failed['id']) == original
        client.post(f"/api/jobs/{retry['id']}/cancel")
        again = client.post(f"/api/jobs/{retry['id']}/retry")
        assert again.status_code == 201 and again.json()['retry_of'] == retry['id']
        client.post(f"/api/jobs/{again.json()['id']}/cancel")
        interrupted = api_submit(client, 'hold-shutdown')
        started(manager, interrupted)
    with TestClient(create_app(tmp_path, command)) as client:
        assert client.get(f"/api/jobs/{interrupted['id']}").json()['status'] == 'interrupted'
        response = client.post(f"/api/jobs/{interrupted['id']}/retry")
        assert response.status_code == 201 and response.json()['retry_of'] == interrupted['id']
        assert client.get(f"/api/jobs/{interrupted['id']}").json()['status'] == 'interrupted'


def test_retry_rejects_live_completed_missing_and_invalid_source(tmp_path):
    input_file = tmp_path / 'input.mp4'
    input_file.write_bytes(b'fixture')
    app = create_app(tmp_path, command, max_queue=1)
    with TestClient(app) as client:
        manager = app.state.manager
        running = api_submit(client)
        queued = api_submit(client, 'hold-queued')
        for job in (running, queued):
            assert client.post(f"/api/jobs/{job['id']}/retry").status_code == 409
        assert client.post('/api/jobs/missing/retry').status_code == 404
        assert client.post('/api/jobs', json=dict(input_path='input.mp4', read_type='video')).status_code == 409
        client.post(f"/api/jobs/{queued['id']}/cancel")
        fill = api_submit(client, 'hold-fill')
        assert client.post(f"/api/jobs/{queued['id']}/retry").status_code == 409
        client.post(f"/api/jobs/{fill['id']}/cancel")
        client.post(f"/api/jobs/{running['id']}/cancel")
        successful = api_submit(client, 'success')
        finished(manager, successful)
        assert client.post(f"/api/jobs/{successful['id']}/retry").status_code == 409
        input_file.unlink()
        assert client.post(f"/api/jobs/{queued['id']}/retry").status_code == 422
        input_file.touch()
        assert client.post(f"/api/jobs/{queued['id']}/retry").status_code == 422
        input_file.write_bytes(b'fixture')
        manager.jobs[queued['id']]['options']['image_sample_interval'] = 0
        assert client.post(f"/api/jobs/{queued['id']}/retry").status_code == 422


def test_normal_completion_reaps_children_before_releasing_leader_pid(tmp_path, monkeypatch):
    import app.jobs as jobs_module
    original_reap = jobs_module.reap_group
    observed_states = []
    def inspect_then_reap(pid):
        # WNOWAIT keeps the exited group leader as a zombie until descendants die.
        observed_states.append(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0])
        original_reap(pid)
    monkeypatch.setattr(jobs_module, 'reap_group', inspect_then_reap)
    with scheduler(tmp_path) as manager:
        job = enqueue(manager, 'tree-success')
        process = started(manager, job)
        release(manager, job)
        finished(manager, job)
        assert observed_states == ['Z']
        assert not alive(process['pid']) and not alive(process['child'])


def test_restart_does_not_signal_an_unrelated_process(tmp_path):
    sleeper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(90)'], start_new_session=True)
    try:
        manager = JobManager(tmp_path, command)
        manager.start()
        job_id = 'b' * 32
        job_dir = manager.jobs_dir / job_id
        job_dir.mkdir()
        atomic_json(job_dir / 'job.json', dict(id=job_id, status='running', stage='running', created_at='2026-10-30'))
        atomic_json(job_dir / 'process.json', dict(pid=sleeper.pid, start='not-this-process-start'))
        manager.close()
        with scheduler(tmp_path) as restarted:
            assert restarted.get(job_id)['status'] == 'interrupted'
            assert sleeper.poll() is None
    finally:
        sleeper.kill()
        sleeper.wait()


def test_api_reports_two_running_jobs_and_dynamic_queue_positions(tmp_path):
    (tmp_path / 'input.mp4').write_bytes(b'fixture')
    app = create_app(tmp_path, command, max_concurrent=2, max_queue=2, gpu_devices='4,5')
    with TestClient(app) as client:
        one, two = api_submit(client, 'hold-one'), api_submit(client, 'hold-two')
        three, four = api_submit(client, 'hold-three'), api_submit(client, 'hold-four')
        state = client.get('/api/system').json()
        assert state['active_job_ids'] == [one['id'], two['id']]
        assert state['active_job_id'] == one['id']
        assert state['scheduler']['gpu_devices'] == ['4', '5']
        assert state['scheduler']['running_count'] == state['scheduler']['queued_count'] == 2
        jobs = {job['id']:job for job in client.get('/api/jobs').json()['jobs']}
        assert jobs[three['id']]['queue_position'] == 1 and jobs[four['id']]['queue_position'] == 2
        client.post(f"/api/jobs/{three['id']}/cancel")
        assert client.get(f"/api/jobs/{four['id']}").json()['queue_position'] == 1
        assert client.get('/api/system').json()['scheduler']['available_queue_slots'] == 1


def test_cli_documents_and_validates_scheduler_flags():
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-m', 'app.server', '--help'], cwd=project,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0
    assert all(flag in result.stdout for flag in ('--max-concurrent', '--max-queue', '--gpu-devices'))
    invalid = subprocess.run([sys.executable, '-m', 'app.server', '--max-concurrent', '2', '--gpu-devices', '0'],
                             cwd=project, capture_output=True, text=True, timeout=15)
    assert invalid.returncode == 2 and 'exceeds the configured GPU pool' in invalid.stderr
