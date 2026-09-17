import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import type { LayerKey } from '../types/api';

/* The three.js scene, owned outside React.
 *
 * React renders the chrome; the scene is a mutable object graph driven at
 * 60 fps by an animation loop. Putting meshes, the camera or the controls into
 * component state would re-render the tree on every orbit frame, so everything
 * here lives in one ref and the component only ever calls methods on it.
 */

/** Ground pad width in world units; `frame()` scales it to the content. */
const GRID_SPAN = 100;
const GRID_DIV = 20;
/** Default framing for the empty scene, in mm. */
const DEFAULT_SPAN = 200;
/** Fixed three-quarter view direction, and the air left around the fit. */
const VIEW_DIR = new THREE.Vector3(0.62, -0.68, 0.48).normalize();
const VIEW_MARGIN = 1.5;
/**
 * Aim this far BELOW the centre, as a fraction of the span. Looking down at a
 * ground plane, the near half spreads across far more of the frame than the
 * far half, so aiming at the geometric centre leaves the plane sitting low.
 * Measured from rendered pixels: -0.13 puts it dead centre, and the
 * relationship is linear, so this holds at every viewport size.
 */
const AIM_LIFT = -0.13;
const AIM_SIDE = 0.0;

/** One material per visual. */
function materials() {
  return {
    part: new THREE.MeshStandardMaterial({
      color: 0x2b6cb0,
      metalness: 0.25,
      roughness: 0.45,
    }),
    tree: new THREE.MeshStandardMaterial({
      color: 0xc8401d,
      metalness: 0.45,
      roughness: 0.35,
    }),
    shell: new THREE.MeshStandardMaterial({
      color: 0xa08a6d,
      metalness: 0.05,
      roughness: 0.85,
      transparent: true,
      opacity: 0.26,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
    /* The fired shell differs from the green one by a fraction of a percent,
       so drawing it solid would just z-fight. It is rendered as a translucent
       back-face skin instead: a wireframe on a marching-cubes mesh is a solid
       mat of triangles at this density and hides everything behind it. */
    fired: new THREE.MeshStandardMaterial({
      color: 0x5b7f9e,
      metalness: 0.05,
      roughness: 0.7,
      transparent: true,
      opacity: 0.3,
      side: THREE.BackSide,
      depthWrite: false,
    }),
  };
}

export type LayerMap = Record<LayerKey, THREE.Object3D | null>;

export interface Scene {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  gizmo: TransformControls;
  root: THREE.Group;
  grid: THREE.GridHelper;
  axes: THREE.AxesHelper;
  layers: LayerMap;
  mat: ReturnType<typeof materials>;
  loader: STLLoader;
  /** Distance at which the whole model just fills the frame — zoom 100%. */
  fitDistance: number | null;
  /** The last framing, so a resize can reapply it. */
  lastSpan: number;
  lastCentre: THREE.Vector3;
  frame: (span: number, centre?: THREE.Vector3) => void;
  fit: () => void;
  dispose: () => void;
}

/** Build the scene and attach it to `host`. */
function createScene(host: HTMLElement): Scene {
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f5f7);

  const grid = new THREE.GridHelper(GRID_SPAN, GRID_DIV, 0x6f7887, 0xa8b0bc);
  grid.rotation.x = Math.PI / 2; // into the z-up frame
  const gridMat = grid.material as THREE.Material;
  gridMat.transparent = true;
  gridMat.opacity = 0.75;
  scene.add(grid);

  const axes = new THREE.AxesHelper(GRID_SPAN * 0.16);
  scene.add(axes);

  const layers: LayerMap = { part: null, tree: null, shell: null, fired: null };
  const root = new THREE.Group();
  scene.add(root);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 20000);
  camera.up.set(0, 0, 1); // z-up: CAD convention

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  /* In three 0.160 TransformControls is itself an Object3D and is added to the
     scene directly. (Later versions split the visuals out behind getHelper();
     if this is ever upgraded, that is the line that changes.) */
  const gizmo = new TransformControls(camera, renderer.domElement);
  scene.add(gizmo as unknown as THREE.Object3D);

  scene.add(new THREE.HemisphereLight(0xffffff, 0xbfc6d0, 1.05));
  const key = new THREE.DirectionalLight(0xffffff, 1.15);
  key.position.set(1, -1.4, 1.6);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xdce6f5, 0.55);
  fill.position.set(-1.2, 1, -0.4);
  scene.add(fill);

  const api: Scene = {
    renderer,
    scene,
    camera,
    controls,
    gizmo,
    root,
    grid,
    axes,
    layers,
    mat: materials(),
    loader: new STLLoader(),
    fitDistance: null,
    lastSpan: DEFAULT_SPAN,
    lastCentre: new THREE.Vector3(),
    frame: () => {},
    fit: () => {},
    dispose: () => {},
  };

  /* Point the camera at `centre` and back off far enough that a sphere of
     diameter `span` fits in the frame. That is all. The orbit target IS the
     centre of the content, so the content is what sits in the middle of the
     viewport and stays there while you orbit. */
  api.frame = (span: number, centre?: THREE.Vector3) => {
    const c = (centre || new THREE.Vector3(0, 0, 0)).clone();
    api.lastSpan = span;
    api.lastCentre = c.clone();

    /* Ground pad and axis marker, centred under the content.
       The pad sits at the BOTTOM of what is drawn, not at world z = 0: `fit()`
       translates the model group so its centre lands on the origin, so a pad
       pinned to z = 0 cuts through the middle of the part and it looks
       half-buried. Track the underside of the visible geometry instead. */
    grid.scale.setScalar(span / GRID_SPAN);
    axes.scale.setScalar(span / GRID_SPAN);

    let floor = c.z - span * 0.5;
    const fb = new THREE.Box3();
    let haveGeom = false;
    for (const k of Object.keys(layers) as LayerKey[]) {
      const m = layers[k];
      if (m && m.visible) {
        fb.expandByObject(m);
        haveGeom = true;
      }
    }
    if (haveGeom && isFinite(fb.min.z)) floor = fb.min.z;

    grid.position.set(c.x, c.y, floor);
    axes.position.set(c.x, c.y, floor);

    // Distance that fits `span` both vertically and horizontally.
    const vFOV = THREE.MathUtils.degToRad(camera.fov);
    const fitH = span / 2 / Math.tan(vFOV / 2);
    const fitW = fitH / Math.max(camera.aspect, 1e-4);
    const dist = Math.max(fitH, fitW) * VIEW_MARGIN;

    const aim = c.clone();
    aim.z += span * AIM_LIFT;
    // Slide the aim point sideways, in the camera's own screen-right direction.
    const fwd = VIEW_DIR.clone().negate();
    const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
    aim.add(right.multiplyScalar(span * AIM_SIDE));

    camera.position.copy(VIEW_DIR.clone().multiplyScalar(dist).add(aim));
    // The distance the zoom picker calls 100%. Reframing moves it, so it is
    // recorded here rather than measured once at start-up.
    api.fitDistance = dist;
    camera.up.set(0, 0, 1); // keep +Z up while orbiting
    camera.near = Math.max(dist / 1000, 0.01);
    camera.far = dist * 100;
    camera.updateProjectionMatrix();
    controls.target.copy(aim);
    controls.update();
  };

  /** Frame everything currently visible. */
  api.fit = () => {
    const box = new THREE.Box3();
    let have = false;
    for (const k of Object.keys(layers) as LayerKey[]) {
      const m = layers[k];
      if (m && m.visible) {
        box.expandByObject(m);
        have = true;
      }
    }
    if (!have || box.isEmpty()) {
      api.frame(DEFAULT_SPAN, new THREE.Vector3());
      return;
    }
    const size = box.getSize(new THREE.Vector3());
    api.frame(Math.max(size.x, size.y, size.z) || DEFAULT_SPAN, box.getCenter(new THREE.Vector3()));
  };

  const resize = () => {
    const w = host.clientWidth;
    const h = host.clientHeight;
    if (!w || !h) return;
    /* The third argument must stay TRUE so three.js also sets the canvas's CSS
       size. With `false` and a pixel ratio of 2 (any Retina display), the
       canvas gets a 2x backing buffer but no CSS size to constrain it, so it
       renders at double the container's size and the scene spills off the
       bottom-right -- which looks exactly like a camera that is not centred. */
    renderer.setSize(w, h, true);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };

  const ro = new ResizeObserver(() => {
    resize();
    api.frame(api.lastSpan, api.lastCentre);
  });
  ro.observe(host);
  resize();
  api.frame(DEFAULT_SPAN, new THREE.Vector3(0, 0, 0));

  let raf = 0;
  const loop = () => {
    raf = requestAnimationFrame(loop);
    controls.update();
    renderer.render(scene, camera);
  };
  loop();

  api.dispose = () => {
    cancelAnimationFrame(raf);
    ro.disconnect();
    controls.dispose();
    gizmo.dispose();
    renderer.dispose();
    renderer.domElement.remove();
  };

  return api;
}

/**
 * Mount the scene into `hostRef` for the life of the component.
 *
 * StrictMode double-invokes effects in development, so the teardown must be
 * complete -- an incomplete dispose leaves a second canvas and a second
 * animation loop running, which looks like a mysterious 2x frame rate.
 */
export function useScene(hostRef: React.RefObject<HTMLElement>) {
  const ref = useRef<Scene | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const s = createScene(host);
    ref.current = s;
    return () => {
      s.dispose();
      ref.current = null;
    };
  }, [hostRef]);

  return ref;
}

export { THREE, DEFAULT_SPAN };
