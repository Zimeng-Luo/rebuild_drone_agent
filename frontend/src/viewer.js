import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

export class SceneViewer {
  constructor(container, onError) {
    this.container = container;
    this.loadVersion = 0;
    this.object = null;
    this.pointSize = 0.025;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#141e21');
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.setAttribute('aria-label', '三维模型视口：拖动旋转，滚轮缩放，右键平移');
    this.renderer.domElement.setAttribute('role', 'img');
    this.renderer.domElement.addEventListener('webglcontextlost', event => {
      event.preventDefault();
      onError('三维渲染上下文已丢失，请刷新页面恢复。');
    });
    container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x56656b, 2.2));
    const light = new THREE.DirectionalLight(0xffffff, 2.5);
    light.position.set(3, 8, 5);
    this.scene.add(light);
    this.grid = new THREE.GridHelper(40, 40, '#54736c', '#2a3a3d');
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.65;
    this.scene.add(this.grid);
    this.reset();
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  resize() {
    const width = Math.max(this.container.clientWidth, 1);
    const height = Math.max(this.container.clientHeight, 1);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  disposeObject(object) {
    object?.traverse(node => {
      node.geometry?.dispose();
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      for (const material of materials) {
        if (!material) continue;
        for (const value of Object.values(material)) if (value?.isTexture) value.dispose();
        material.dispose();
      }
    });
  }

  clear() {
    this.loadVersion++;
    if (this.object) {
      this.scene.remove(this.object);
      this.disposeObject(this.object);
    }
    this.object = null;
    this.grid.scale.setScalar(1);
    this.grid.position.set(0, 0, 0);
    this.reset();
  }

  async load(url, kind) {
    this.clear();
    const version = this.loadVersion;
    let object;
    if (kind === 'ply') {
      const geometry = await new PLYLoader().loadAsync(url);
      geometry.computeBoundingBox();
      object = new THREE.Points(geometry, new THREE.PointsMaterial({
        size: this.pointSize, sizeAttenuation: true,
        vertexColors: geometry.hasAttribute('color'),
        color: geometry.hasAttribute('color') ? 0xffffff : 0xc9e4d9
      }));
    } else {
      object = (await new GLTFLoader().loadAsync(url)).scene;
    }
    if (version !== this.loadVersion) { this.disposeObject(object); return false; }
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) { this.disposeObject(object); throw new Error('模型不包含可显示的几何数据。'); }
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    if (![...center, ...size].every(Number.isFinite)) { this.disposeObject(object); throw new Error('模型坐标无效，无法显示。'); }
    object.position.sub(center);
    this.object = object;
    this.scene.add(object);
    const extent = Math.max(size.x, size.y, size.z, 0.001);
    this.grid.scale.setScalar(extent / 15);
    this.grid.position.y = -size.y / 2 - extent * 0.002;
    this.setPointSize(this.pointSize);
    this.reset();
    return true;
  }

  reset() {
    const box = this.object ? new THREE.Box3().setFromObject(this.object) : null;
    const extent = box ? Math.max(...box.getSize(new THREE.Vector3()), 0.001) : 12;
    const fitFov = Math.min(this.camera.fov, this.camera.fov * Math.max(this.camera.aspect, 0.2));
    const distance = extent / (2 * Math.tan(THREE.MathUtils.degToRad(fitFov / 2))) * 1.6;
    this.camera.position.set(distance * 0.72, distance * 0.55, distance * 0.9);
    this.camera.near = Math.max(extent / 10000, 0.00001);
    this.camera.far = Math.max(extent * 10000, 1000);
    this.camera.updateProjectionMatrix();
    this.controls.target.set(0, 0, 0);
    this.controls.minDistance = extent / 1000;
    this.controls.maxDistance = extent * 100;
    this.controls.update();
  }

  setPointSize(size) {
    this.pointSize = Number(size);
    this.object?.traverse(node => {
      if (node.isPoints && node.material) {
        const materials = Array.isArray(node.material) ? node.material : [node.material];
        materials.forEach(material => { material.size = this.pointSize; });
      }
    });
  }
  setGrid(visible) { this.grid.visible = visible; }
  setLight(light) { this.scene.background = new THREE.Color(light ? '#e8ede9' : '#141e21'); }
  dispose() {
    this.clear();
    this.observer.disconnect();
    this.controls.dispose();
    this.grid.geometry.dispose();
    this.grid.material.dispose();
    this.renderer.setAnimationLoop(null);
    this.renderer.dispose();
  }
}
