"""API integration tests use actual small subprocesses, never reconstruction models."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from app.jobs import JobManager, atomic_json, tail
from app.server import create_app

PROJECT = Path(__file__).resolve().parents[1]
FIXTURE_CODE = r'''
import json, os, pathlib, signal, subprocess, sys, time
folder = pathlib.Path(sys.argv[1])
request = json.loads((folder / 'request.json').read_text())
job = json.loads((folder / 'job.json').read_text())
mode = job['name']
print('[workbench-stage] rebuilding', flush=True)
if mode in ('sleep', 'shutdown'):
    child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(90)'])
    (folder / 'child.pid').write_text(str(child.pid))
    time.sleep(90)
if mode == 'failure':
    print('fixture exploded with useful reason', flush=True)
    (folder / 'result.json').write_text(json.dumps({'status':'failed','error':'fixture reconstruction failure'}))
    sys.exit(7)
if mode == 'no_result':
    sys.exit(0)
output = pathlib.Path(request['result_dir']) / '2026-10-29_200000'
output.mkdir()
(output / 'aligned.glb').write_bytes(b'glTF-fixture')
(output / 'aligned.ply').write_text('ply\nformat ascii 1.0\nend_header\n')
(output / 'pipeline.log').write_text('real fixture log')
(output / 'task_summary.log').write_text(json.dumps({'time':{'total':{'duration_seconds':0.01}}}))
(output / 'secret.pkl').write_bytes(b'not-public')
(output / 'raw.npz').write_bytes(b'not-public')
(output / 'escape.json').symlink_to(folder / 'request.json')
print('fixture complete', flush=True)
(folder / 'result.json').write_text(json.dumps({'status':'completed','summary':{'fixture':True}}))
'''


def command(folder):
    return [sys.executable, '-u', '-c', FIXTURE_CODE, str(folder)]


@pytest.fixture
def workspace(tmp_path):
    video = tmp_path / 'input.mp4'
    video.write_bytes(b'fixture-video')
    return tmp_path


@pytest.fixture
def client(workspace):
    app = create_app(workspace, worker_command_factory=command)
    with TestClient(app) as session:
        yield session


def submit(client, workspace, name='success', **kwargs):
    return client.post('/api/jobs', json=dict(input_path=str(workspace / 'input.mp4'), read_type='video', name=name, **kwargs))


def wait_job(client, job_id, status=None):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f'/api/jobs/{job_id}').json()
        if status and job['stage'] == status:
            return job
        if not status and job['status'] not in {'queued', 'running'}:
            return job
        time.sleep(0.03)
    pytest.fail(f'Job never reached expected state: {job}')


def child_is_running(pid):
    stat = Path(f'/proc/{pid}/stat')
    return stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z'


def wait_child_pid(workspace, job_id):
    path = workspace / 'data/workbench/jobs' / job_id / 'child.pid'
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return int(path.read_text())


def test_import_is_lightweight():
    code = 'import sys; import app.server; assert "torch" not in sys.modules; assert "main" not in sys.modules; assert "pipeline" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], cwd=PROJECT, check=True, timeout=30)


def test_system_static_fallback_and_api_404(client):
    system = client.get('/api/system').json()
    assert set(system) == {'version', 'gpu', 'disk', 'active_job_id', 'active_job_ids', 'scheduler'}
    assert system['active_job_ids'] == [] and system['active_job_id'] is None
    assert system['scheduler'] == dict(max_concurrent=1, max_queue=8, running_count=0,
                                       queued_count=0, available_queue_slots=8, gpu_devices=[],
                                       ready=True, configuration_error=None)
    assert 'available' in system['gpu'] and system['disk']['free_gb'] > 0
    response = client.get('/')
    assert response.status_code == 503 and 'npm run build' in response.text
    assert client.get('/api/not-found').status_code == 404


def test_static_containment(workspace):
    static = workspace / 'app/static'
    static.mkdir(parents=True)
    (static / 'index.html').write_text('<html>fixture</html>')
    (workspace / 'private.json').write_text('{"private":true}')
    (static / 'escape.json').symlink_to(workspace / 'private.json')
    with TestClient(create_app(workspace, command)) as client:
        assert client.get('/').text == '<html>fixture</html>'
        assert client.get('/escape.json').status_code == 404
        assert client.get('/%2e%2e/private.json').status_code == 404


def test_upload_video_and_safe_basename(client, workspace):
    response = client.post('/api/uploads', files=[('files', ('../../nested/test.MP4', b'video', 'video/mp4'))])
    assert response.status_code == 200, response.text
    upload = response.json()
    assert upload['read_type'] == 'video'
    path = Path(upload['input_path'])
    assert path.name == 'test.MP4' and path.is_relative_to(workspace / 'data/uploads')
    assert path.read_bytes() == b'video'
    assert not (workspace / 'test.MP4').exists()


def test_upload_images(client):
    response = client.post('/api/uploads', files=[('files', ('a.jpg', b'image-a')), ('files', ('b.png', b'image-b'))])
    assert response.status_code == 200
    payload = response.json()
    assert payload['read_type'] == 'image' and Path(payload['input_path']).is_dir()


@pytest.mark.parametrize('files,reason', [
    ([('files', ('empty.mp4', b''))], 'Empty'),
    ([('files', ('one.jpg', b'a'))], 'at least 2'),
    ([('files', ('one.mp4', b'a')), ('files', ('two.mp4', b'b'))], 'exactly 1'),
    ([('files', ('one.mp4', b'a')), ('files', ('two.jpg', b'b'))], 'mix'),
    ([('files', ('one.py', b'a'))], 'Unsupported'),
    ([('files', ('x/a.jpg', b'a')), ('files', ('y/a.jpg', b'b'))], 'Duplicate'),
])
def test_reject_bad_uploads(client, workspace, files, reason):
    response = client.post('/api/uploads', files=files)
    assert response.status_code == 422 and reason in response.text
    uploads = workspace / 'data/uploads'
    assert not uploads.exists() or list(uploads.iterdir()) == []


def test_upload_limits_cleanup(client, workspace, monkeypatch):
    import app.server as server
    monkeypatch.setattr(server, 'MAX_UPLOAD_BYTES', 5)
    response = client.post('/api/uploads', files=[('files', ('one.mp4', b'123456'))])
    assert response.status_code == 413
    assert list((workspace / 'data/uploads').iterdir()) == []
    monkeypatch.setattr(server, 'MAX_UPLOAD_BYTES', 1024)
    monkeypatch.setattr(server, 'MAX_IMAGE_BYTES', 2)
    response = client.post('/api/uploads', files=[('files', ('a.jpg', b'123')), ('files', ('b.jpg', b'12'))])
    assert response.status_code == 413 and '100 MiB' in response.text


@pytest.mark.parametrize('changes', [
    {'read_type':'wrong'}, {'input_path':'missing.mp4'}, {'image_sample_interval':0},
    {'image_sample_interval':True}, {'confidence_percentile':101}, {'confidence_percentile':-1},
    {'remove_mask_enable':'true'}, {'unexpected':'field'}, {'name':'x'*201},
])
def test_invalid_job_requests(client, workspace, changes):
    body = dict(input_path=str(workspace / 'input.mp4'), read_type='video')
    body.update(changes)
    assert client.post('/api/jobs', json=body).status_code == 422


def test_input_type_empty_and_images_validation(client, workspace):
    empty = workspace / 'empty.mp4'
    empty.touch()
    assert client.post('/api/jobs', json={'input_path':str(empty), 'read_type':'video'}).status_code == 422
    folder = workspace / 'images'
    folder.mkdir()
    (folder / 'a.jpg').write_bytes(b'a')
    assert client.post('/api/jobs', json={'input_path':str(folder), 'read_type':'image'}).status_code == 422
    (folder / 'b.jpg').write_bytes(b'b')
    (folder / 'mixed.mp4').write_bytes(b'x')
    assert client.post('/api/jobs', json={'input_path':str(folder), 'read_type':'image'}).status_code == 422


def test_success_persistence_artifacts_logs(client, workspace):
    response = submit(client, workspace)
    assert response.status_code == 201, response.text
    job_id = response.json()['id']
    job = wait_job(client, job_id)
    assert job['status'] == 'completed' and job['summary'] == {'fixture':True}
    assert job['error'] is None and job['started_at'] and job['finished_at']
    assert job['options']['image_sample_interval'] == 3
    names = [item['name'] for item in job['outputs']]
    assert len(names) == 4 and all(not name.endswith(('.pkl', '.npz')) for name in names)
    artifact = next(item for item in job['outputs'] if item['name'].endswith('.glb'))
    assert client.get(artifact['url']).content == b'glTF-fixture'
    prefix = f'/api/jobs/{job_id}/artifacts/'
    assert client.get(prefix + '2026-10-29_200000/secret.pkl').status_code == 404
    assert client.get(prefix + '2026-10-29_200000/escape.json').status_code == 404
    assert client.get(prefix + '%2e%2e/request.json').status_code == 404
    logs = client.get(f'/api/jobs/{job_id}/logs?limit=1').json()['lines']
    assert logs == ['fixture complete']
    assert client.get(f'/api/jobs/{job_id}/logs?limit=1001').status_code == 422
    persisted = json.loads((workspace / 'data/workbench/jobs' / job_id / 'job.json').read_text())
    assert persisted['status'] == 'completed' and persisted['outputs'] == job['outputs']
    assert client.get('/api/jobs').json()['jobs'][0]['id'] == job_id
    assert client.get('/api/jobs/unknown').status_code == 404
    # A previously enumerated product replaced by an external symlink is rejected.
    disk_output = workspace / 'data/result/workbench' / job_id / artifact['name']
    disk_output.unlink()
    disk_output.symlink_to(workspace / 'input.mp4')
    assert client.get(artifact['url']).status_code == 404


@pytest.mark.parametrize('mode,reason', [('failure','fixture reconstruction failure'), ('no_result','without a valid result')])
def test_worker_failure_never_completed(client, workspace, mode, reason):
    job = wait_job(client, submit(client, workspace, mode).json()['id'])
    assert job['status'] == 'failed' and reason in job['error']
    assert job['finished_at']


def test_single_running_stage_and_cancel_group(client, workspace):
    job_id = submit(client, workspace, 'sleep').json()['id']
    child_pid = wait_child_pid(workspace, job_id)
    assert child_is_running(child_pid)
    assert wait_job(client, job_id, 'rebuilding')['status'] == 'running'
    queued = submit(client, workspace)
    assert queued.status_code == 201 and queued.json()['queue_position'] == 1
    assert queued.json()['status'] == 'queued' and queued.json()['gpu_device'] is None
    assert client.get('/api/system').json()['active_job_id'] == job_id
    job = client.post(f'/api/jobs/{job_id}/cancel').json()
    assert job['status'] == 'cancelled' and job['finished_at']
    assert not child_is_running(child_pid)
    assert client.post(f'/api/jobs/{job_id}/cancel').json()['status'] == 'cancelled'
    assert wait_job(client, queued.json()['id'])['status'] == 'completed'
    assert client.get('/api/system').json()['active_job_id'] is None
    assert wait_job(client, submit(client, workspace).json()['id'])['status'] == 'completed'


def test_shutdown_interrupts_group_and_restart_preserves(workspace):
    with TestClient(create_app(workspace, command)) as client:
        job_id = submit(client, workspace, 'shutdown').json()['id']
        child_pid = wait_child_pid(workspace, job_id)
    assert not child_is_running(child_pid)
    with TestClient(create_app(workspace, command)) as client:
        job = client.get(f'/api/jobs/{job_id}').json()
        assert job['status'] == 'interrupted' and 'shutdown' in job['error']
        assert client.get('/api/system').json()['active_job_id'] is None


@pytest.mark.parametrize('status', ['queued', 'running'])
def test_restart_marks_abandoned_active_job(workspace, status):
    manager = JobManager(workspace, command)
    manager.start()
    job_id = 'a' * 32
    folder = manager.jobs_dir / job_id
    folder.mkdir()
    atomic_json(folder / 'job.json', dict(id=job_id, status=status, stage=status, created_at='2026-10-29T00:00:00+00:00'))
    manager.close()
    with TestClient(create_app(workspace, command)) as client:
        job = client.get(f'/api/jobs/{job_id}').json()
        assert job['status'] == 'interrupted' and 'restarted' in job['error'] and job['finished_at']


def test_second_manager_rejected(workspace):
    first = JobManager(workspace, command)
    first.start()
    second = JobManager(workspace, command)
    try:
        with pytest.raises(RuntimeError, match='already owns'):
            second.start()
    finally:
        first.close()


def test_discovery_stops_at_image_directory(client, workspace):
    base = workspace / 'data/example'
    nested = base / 'dataset/site/images'
    nested.mkdir(parents=True)
    (nested / '01.jpg').write_bytes(b'1')
    (nested / '02.jpg').write_bytes(b'2')
    (base / 'demo.MP4').write_bytes(b'video')
    items = client.get('/api/inputs').json()['items']
    assert {item['type'] for item in items} == {'image','video'}
    assert any(item['path'] == str(nested) for item in items)


def test_no_arbitrary_input_file_reader_and_cross_origin(client, workspace):
    assert client.get('/api/inputs?path=/etc/passwd').status_code == 200
    assert client.get('/etc/passwd').status_code == 404
    assert client.post('/api/jobs', json={'input_path':str(workspace / 'input.mp4'),'read_type':'video'}, headers={'Origin':'https://external.example'}).status_code == 403


def test_tail_bounded(workspace):
    path = workspace / 'large.log'
    path.write_text('x' * 100000 + '\nlast\n')
    assert tail(path, max_lines=1, max_bytes=30) == ['last']


def test_discovery_scattered_root_images_do_not_hide_videos(client, workspace):
    base = workspace / 'data/example'
    base.mkdir(parents=True)
    (base / 'a.png').write_bytes(b'1')
    (base / 'b.png').write_bytes(b'2')
    (base / 'demo.mp4').write_bytes(b'video')
    items = client.get('/api/inputs').json()['items']
    assert items == [{'path':str(base / 'demo.mp4'), 'name':'demo.mp4', 'type':'video'}]


def test_worker_adapter_maps_pipeline_arguments(workspace, monkeypatch):
    import types
    import app.worker as worker
    folder = workspace / 'worker-job'
    folder.mkdir()
    result_dir = workspace / 'results'
    calls = {}
    fake = types.ModuleType('main')
    fake.run_data_preprocess_with_task_dir = lambda: calls.update(preprocess=True)
    fake.run_rebuild_chunks = lambda: calls.update(rebuild=True)
    fake.run_align_and_merge = lambda: calls.update(align=True)
    def pipeline(**kwargs):
        calls.update(kwargs)
        fake.run_data_preprocess_with_task_dir()
        fake.run_rebuild_chunks()
        fake.run_align_and_merge()
        output = Path(kwargs['result_dir']) / 'timestamp/aligned.glb'
        output.parent.mkdir(parents=True)
        output.write_bytes(b'glTF')
        (output.parent / 'task_summary.log').write_text('{"measured":true}')
        return output
    fake.run_pipeline = pipeline
    monkeypatch.setitem(sys.modules, 'main', fake)
    monkeypatch.setattr(worker, 'protect_parent', lambda _: None)
    atomic_json(folder / 'request.json', dict(parent_pid=os.getpid(), input_path=str(workspace / 'input.mp4'),
                read_type='video', result_dir=str(result_dir), options=dict(image_sample_interval=7,
                remove_mask_enable=True, confidence_percentile=23.5, downsample_enable=True)))
    assert worker.run(folder) == 0
    assert calls['image_sample_interval'] == 7 and calls['remove_mask_enable'] is True
    assert calls['align_glb_conf_thres'] == calls['align_ply_conf_thres'] == 23.5
    assert calls['pointcloud_downsample_enable'] is True
    assert calls['export_glb_enable'] is calls['export_ply_enable'] is True
    assert calls['preprocess'] and calls['rebuild'] and calls['align']
    result = json.loads((folder / 'result.json').read_text())
    assert result['status'] == 'completed' and result['summary'] == {'measured':True}


def test_worker_adapter_records_exception(workspace, monkeypatch):
    import types
    import app.worker as worker
    folder = workspace / 'bad-worker-job'
    folder.mkdir()
    fake = types.ModuleType('main')
    for name in ('run_data_preprocess_with_task_dir', 'run_rebuild_chunks', 'run_align_and_merge'):
        setattr(fake, name, lambda: None)
    def fail(**kwargs):
        raise RuntimeError('fixture actual pipeline failure')
    fake.run_pipeline = fail
    monkeypatch.setitem(sys.modules, 'main', fake)
    monkeypatch.setattr(worker, 'protect_parent', lambda _: None)
    atomic_json(folder / 'request.json', dict(parent_pid=os.getpid(), input_path='input.mp4', read_type='video',
                result_dir=str(workspace / 'results'), options=dict(image_sample_interval=3,
                remove_mask_enable=False, confidence_percentile=0, downsample_enable=False)))
    assert worker.run(folder) == 1
    result = json.loads((folder / 'result.json').read_text())
    assert result == {'status':'failed', 'error':'RuntimeError: fixture actual pipeline failure'}


def test_parent_death_terminates_worker_and_descendant(workspace):
    # Execute the real worker's lightweight parent-death setup in a new session.
    worker_code = '''import os,pathlib,subprocess,sys,time
from app.worker import protect_parent
protect_parent(os.getppid())
child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(90)'])
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()) + ',' + str(child.pid))
time.sleep(90)
'''
    marker = workspace / 'parent-death-pids'
    controller_code = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2]],start_new_session=True);time.sleep(90)'
    parent = subprocess.Popen([sys.executable, '-c', controller_code, worker_code, str(marker)], cwd=PROJECT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    worker_pid = None
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        worker_pid, child_pid = map(int, marker.read_text().split(','))
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while (child_is_running(worker_pid) or child_is_running(child_pid)) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not child_is_running(worker_pid) and not child_is_running(child_pid)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if worker_pid is not None:
            try:
                os.killpg(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_upload_without_content_length_is_bounded(client, monkeypatch):
    import app.server as server
    monkeypatch.setattr(server, 'MAX_UPLOAD_BYTES', 5)
    def body():
        yield b'--boundary\r\nContent-Disposition: form-data; name="files"; filename="large.mp4"\r\n\r\n'
        for _ in range(17):
            yield b'x' * 1024**2
        yield b'\r\n--boundary--\r\n'
    response = client.post('/api/uploads', content=body(), headers={'Content-Type':'multipart/form-data; boundary=boundary'})
    assert response.status_code == 413 and 'total limit' in response.text
