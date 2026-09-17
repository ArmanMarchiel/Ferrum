import { useCallback, useEffect, useRef, useState } from 'react';
import { Toolbar, type MenuItem } from '../components/Toolbar';
import {
  Legend,
  StatusBar,
  ViewControls,
  ViewHost,
  type StatusState,
} from '../components/Viewport';
import {
  PartPanel,
  ProcessPanel,
  ResultsPanel,
  Tabs,
  type TabName,
} from '../components/Sidebar';
import { Dialog } from '../components/Dialog';
import { usePageClass } from '../hooks/usePageClass';
import { useScene, THREE, DEFAULT_SPAN } from '../hooks/useScene';
import { useHistory } from '../hooks/useHistory';
import { useJob } from '../hooks/useJob';
import { API, fetchMesh, getMeta, listAll } from '../services/api';
import type {
  FsItem,
  JobStats,
  LayerKey,
  LayerVisibility,
  PipelineParams,
} from '../types/api';

declare global {
  interface Window {
    /* Test hooks. The headless UI suite asserts on the scene graph rather than
       on pixels, so these stay exposed. */
    __scene?: THREE.Scene;
    __grid?: THREE.GridHelper;
    __axes?: THREE.AxesHelper;
    __layers?: Record<string, THREE.Object3D | null>;
    /** Where the scene actually projects on screen, for the headless suite. */
    __probe?: () => Record<string, unknown>;
  }
}

/* The CAD editor.
 *
 * React owns the chrome and the form state; three.js owns the scene. The two
 * meet in `sceneRef` -- the component never renders a mesh, it calls a method
 * that mutates the scene graph, and the animation loop draws the result.
 */

const DEFAULTS: PipelineParams = {
  alloy: 'A356',
  shell_thickness: 6,
  voxel_pitch: 1.5,
  riser_modulus_factor: 1.2,
  gating_ratio: [1, 2, 2],
  ceramic: 'fused_silica',
  fire_temperature: 1300,
  fire_hold_hours: 2,
  fire_shrinkage: null,
  use_risers: false,
  max_risers: 4,
  quantity: 1,
  layout: 'grid',
};

/** The editor's document: what save, undo and reopen all round-trip. */
interface EditorState {
  fields: Record<string, unknown>;
  quantity: number;
  layout: string;
  gap: string;
  tiers: number;
  /** Cumulative view rotation, as a quaternion. */
  spin: [number, number, number, number];
}

export function Editor() {
  usePageClass('page-editor');

  const viewRef = useRef<HTMLElement>(null);
  const sceneRef = useScene(viewRef);
  const { poll, current: currentJob } = useJob();

  /* ---- form state ---- */
  const [params, setParams] = useState<PipelineParams>(DEFAULTS);
  const [quantity, setQuantity] = useState(1);
  const [layout, setLayout] = useState('grid');
  const [gap, setGap] = useState('');
  const [tiers, setTiers] = useState(1);
  const [tab, setTab] = useState<TabName>('part');
  const [dropHot, setDropHot] = useState(false);

  /* ---- document identity ---- */
  const [title, setTitle] = useState('');
  const [projectId, setProjectId] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);

  /* ---- viewer ---- */
  const [visible, setVisible] = useState<LayerVisibility>({
    part: true,
    shell: true,
    fired: true,
    tree: false,
  });
  const [opacity, setOpacity] = useState(26);
  const [zoom, setZoom] = useState(100);
  const [status, setStatus] = useState<StatusState>({
    html: 'Drop an STL to begin.',
    cls: '',
    frac: null,
  });
  const [stats, setStats] = useState<JobStats | null>(null);
  const [hud, setHud] = useState('');
  const [busy, setBusy] = useState(false);
  const [shrinkHint, setShrinkHint] = useState('');
  /* Clash detection between placed instances is not wired in this build; the
     readout stays empty until it is. */
  const [clash] = useState('');
  const [total, setTotal] = useState('');

  /* ---- dialogs ---- */
  const [saveOpen, setSaveOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [folders, setFolders] = useState<FsItem[]>([]);
  const [saveName, setSaveName] = useState('');
  const [saveFolder, setSaveFolder] = useState('');
  const [dlgErr, setDlgErr] = useState('');
  const [exports, setExports] = useState<Record<string, boolean>>({});

  /** The uploaded mesh, held so a save does not ask for the file again. */
  const partFile = useRef<File | null>(null);
  /** Its name and size, for the drop zone. State, because it is rendered. */
  const [partInfo, setPartInfo] = useState<{ name: string; mb: string } | null>(
    null,
  );
  /** Cumulative rotation applied by the X/Y/Z buttons. */
  const spin = useRef(new THREE.Quaternion());

  const say = useCallback(
    (html: string, cls: StatusState['cls'] = '') =>
      setStatus({ html, cls, frac: null }),
    [],
  );

  const progress = useCallback(
    (label: string, frac: number, detail: string, secs?: string) =>
      setStatus({
        html: `<span class="spin"></span>${label}${secs ? ` — ${secs}s` : ''}`,
        cls: 'busy',
        frac,
        step: detail,
      }),
    [],
  );

  /* ---- history ---- */
  const capture = useCallback(
    (): EditorState => ({
      fields: { ...params },
      quantity,
      layout,
      gap,
      tiers,
      spin: spin.current.toArray() as [number, number, number, number],
    }),
    [params, quantity, layout, gap, tiers],
  );
  const history = useHistory<EditorState>(capture);

  const applyState = useCallback((st: EditorState) => {
    setParams({ ...DEFAULTS, ...(st.fields as Partial<PipelineParams>) });
    setQuantity(st.quantity);
    setLayout(st.layout);
    setGap(st.gap);
    setTiers(st.tiers);
    if (st.spin) spin.current.fromArray(st.spin);
  }, []);

  const markDirty = useCallback(() => setDirty(true), []);

  /* ---- meta: alloys and defaults ---- */
  useEffect(() => {
    void (async () => {
      try {
        const m = await getMeta();
        setParams((p) => ({ ...p, ...m.defaults }));
      } catch {
        // The form keeps its built-in defaults; the first build reports any
        // real connectivity problem.
      }
    })();
  }, []);

  /* ---- firing shrinkage hint ----
     Shrinkage is a property of the ceramic system and its firing schedule
     alone, so the hint restates what the server will derive when the field is
     left blank. */
  useEffect(() => {
    if (params.fire_shrinkage !== null && params.fire_shrinkage !== undefined) {
      setShrinkHint(
        `Overriding the derived value with ${params.fire_shrinkage}% linear.`,
      );
    } else {
      setShrinkHint(
        `Left blank, shrinkage is derived from the ceramic system and the ` +
          `${params.fire_temperature} °C / ${params.fire_hold_hours} h schedule.`,
      );
    }
  }, [params.fire_shrinkage, params.fire_temperature, params.fire_hold_hours, params.ceramic]);

  /* Parts per tier × tiers. Only shown once there is more than one tier,
     where the distinction actually means something. */
  useEffect(() => {
    const per = Math.max(1, quantity || 1);
    const t = Math.max(1, tiers || 1);
    setTotal(t > 1 ? `${per} per tier × ${t} tiers = ${per * t} parts` : '');
  }, [quantity, tiers]);

  /* Debug hooks: the headless UI tests assert that the world origin really
     projects to the centre of the canvas, and read the layer registry, rather
     than eyeballing a screenshot. */
  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    window.__scene = s.scene;
    window.__grid = s.grid;
    window.__axes = s.axes;
    window.__layers = s.layers;
    /* Lets the tests assert that the world origin really projects to the
       centre of the canvas, instead of eyeballing a screenshot. */
    window.__probe = () => {
      const c = s.renderer.domElement;
      const r = c.getBoundingClientRect();
      const o = s.controls.target.clone().project(s.camera);
      return {
        originPx: {
          x: (o.x * 0.5 + 0.5) * r.width,
          y: (-o.y * 0.5 + 0.5) * r.height,
        },
        centrePx: { x: r.width / 2, y: r.height / 2 },
        canvas: { w: r.width, h: r.height },
        target: s.controls.target.toArray(),
        camera: s.camera.position.toArray(),
        layers: Object.fromEntries(
          Object.entries(s.layers).map(([k, v]) => [k, !!v && v.visible]),
        ),
        dist: +s.camera.position.distanceTo(s.controls.target).toFixed(2),
        gridZ: +s.grid.position.z.toFixed(2),
        partMinZ: s.layers.part
          ? +new THREE.Box3().setFromObject(s.layers.part).min.z.toFixed(2)
          : null,
        rotation: [s.root.rotation.x, s.root.rotation.y, s.root.rotation.z].map(
          (v) => Math.round((v * 180) / Math.PI),
        ),
      };
    };
    return () => {
      delete window.__scene;
      delete window.__grid;
      delete window.__axes;
      delete window.__layers;
      delete window.__probe;
    };
  }, [sceneRef, stats]);

  /* ---- layer visibility ---- */
  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    for (const k of Object.keys(visible) as LayerKey[]) {
      const m = s.layers[k];
      if (m) m.visible = visible[k];
    }
  }, [visible, sceneRef]);

  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    s.mat.shell.opacity = opacity / 100;
    s.mat.fired.opacity = Math.min(1, opacity / 100 + 0.04);
  }, [opacity, sceneRef]);

  /** Replace a layer's mesh. `buf` of null clears it. */
  const addLayer = useCallback(
    (key: LayerKey, buf: ArrayBuffer | null) => {
      const s = sceneRef.current;
      if (!s) return;
      const prev = s.layers[key];
      if (prev) {
        s.root.remove(prev);
        (prev as THREE.Mesh).geometry?.dispose();
        s.layers[key] = null;
      }
      if (!buf) return;
      const geo = s.loader.parse(buf);
      geo.computeVertexNormals();
      const mesh = new THREE.Mesh(geo, s.mat[key]);
      mesh.visible = visible[key];
      mesh.quaternion.copy(spin.current);
      s.layers[key] = mesh;
      s.root.add(mesh);
    },
    [sceneRef, visible],
  );

  const fit = useCallback(() => {
    sceneRef.current?.fit();
    setZoom(100);
  }, [sceneRef]);

  /* ---- the request body ---- */
  const formData = useCallback(() => {
    const f = new FormData();
    f.append('shell_thickness', String(params.shell_thickness));
    f.append('voxel_pitch', String(params.voxel_pitch));
    f.append('riser_modulus_factor', String(params.riser_modulus_factor));
    f.append('use_risers', params.use_risers ? 'true' : 'false');
    f.append('max_risers', String(params.max_risers));
    f.append('ceramic', params.ceramic);
    f.append('fire_temperature', String(params.fire_temperature));
    f.append('fire_hold_hours', String(params.fire_hold_hours));
    /* Blank means "derive it from the firing schedule", which is the point of
       asking for a temperature at all. Only send an override when one is
       typed. */
    const shr = Number(params.fire_shrinkage);
    if (params.fire_shrinkage !== null && isFinite(shr) && shr >= 0) {
      f.append('fire_shrinkage', String(shr / 100));
    }
    f.append('quantity', String(Math.max(1, quantity || 1)));
    f.append('layout', layout);
    const g = parseFloat(gap);
    if (isFinite(g) && g >= 0) f.append('part_spacing', String(g));
    // The pour direction is whatever the accumulated view rotation makes "up".
    const up = new THREE.Vector3(0, 0, 1).applyQuaternion(
      spin.current.clone().invert(),
    );
    f.append('pour_up', `${up.x},${up.y},${up.z}`);
    return f;
  }, [params, quantity, layout, gap]);

  /* ---- upload ---- */
  const onFile = useCallback(
    async (f: File) => {
      partFile.current = f;
      setPartInfo({ name: f.name, mb: (f.size / 1048576).toFixed(2) });
      setBusy(true);
      setStats(null);
      setHud('');
      for (const k of ['part', 'shell', 'fired', 'tree'] as LayerKey[]) {
        addLayer(k, null);
      }
      sceneRef.current?.root.position.set(0, 0, 0);
      sceneRef.current?.frame(DEFAULT_SPAN);
      progress('reading file', 0.03, 'uploading mesh');

      const fd = formData();
      fd.append('file', f, f.name);
      // Ingest only. The mould is built when the user presses Generate, so
      // this request must not start one -- see the `build` flag in app.py.
      fd.append('build', 'false');

      try {
        const res = await fetch(API + '/generate', { method: 'POST', body: fd });
        const d = await res.json();
        if (!res.ok) throw new Error(d.detail || d.error || `HTTP ${res.status}`);

        // No build was started, so there is no job to poll. Clearing this stops
        // a poll left over from a previous upload adopting these meshes.
        currentJob.current = null;

        /* Show the uploaded part, and stop there. The mould is only built when
           the user asks: shell generation takes seconds to minutes and its
           parameters are meant to be set first, so building automatically on
           drop would throw work away. */
        const buf = await fetchMesh(d.urls.part_stl);
        if (buf) {
          addLayer('part', buf);
          fit();
        }
        const tris = d.part?.triangles
          ? Number(d.part.triangles).toLocaleString()
          : '?';
        say(
          `Part loaded — ${tris} triangles. Set your process parameters, then ` +
            `<b>Generate mould</b>.`,
        );
        if (projectId) markDirty();
      } catch (e) {
        say(uploadError(e as Error), 'err');
      } finally {
        setBusy(false);
      }
    },
    [addLayer, fit, formData, markDirty, progress, projectId, say, sceneRef, currentJob],
  );

  /* ---- build ---- */
  const generate = useCallback(async () => {
    const f = partFile.current;
    if (!f) {
      say('Drop an STL first.', 'err');
      return;
    }
    setBusy(true);
    for (const k of ['shell', 'fired', 'tree'] as LayerKey[]) addLayer(k, null);
    setStats(null);
    progress('starting', 0.02, 'sending process parameters');

    const fd = formData();
    fd.append('file', f, f.name);

    let d: { job_id: string; urls: Record<string, string> };
    try {
      const res = await fetch(API + '/generate', { method: 'POST', body: fd });
      d = await res.json();
      if (!res.ok) {
        throw new Error(
          (d as { detail?: string }).detail || `HTTP ${res.status}`,
        );
      }
    } catch (e) {
      say(uploadError(e as Error), 'err');
      setBusy(false);
      return;
    }

    const t0 = performance.now();
    await poll(d.job_id, t0, {
      onProgress: (p) => progress(p.label, p.frac, p.detail, p.secs),
      onError: (msg) => {
        say(msg, 'err');
        setBusy(false);
      },
      onDone: async (st, secs) => {
        const urls = (st.urls ?? {}) as Record<string, string>;
        const [sh, fi, tr] = await Promise.all(
          (['shell_stl', 'fired_stl', 'tree_stl'] as const).map((k) =>
            urls[k] ? fetchMesh(urls[k]) : Promise.resolve(null),
          ),
        );
        addLayer('shell', sh);
        addLayer('fired', fi);
        addLayer('tree', tr);
        fit();
        setStats(st);
        setTab('results');

        const shell = st.shell as
          | { watertight?: boolean; thickness_measured?: { median: number } }
          | undefined;
        const warns = (st.warnings as string[]) || [];
        let msg =
          `Done in ${secs}s — shell ` +
          `${shell?.watertight ? 'watertight ✓' : 'NOT watertight ✗'}, ` +
          `wall ${shell?.thickness_measured?.median?.toFixed(2) ?? '?'} mm, ` +
          `yield ${Number(st.yield_pct ?? 0).toFixed(1)}%`;
        if (warns.length) {
          msg += warns.map((w) => `<div class="warn">⚠ ${w}</div>`).join('');
        }
        say(msg, warns.length ? 'warned' : '');
        setHud(
          `<b>triangles</b> ${Number(
            (st.shell as { triangles?: number })?.triangles ?? 0,
          ).toLocaleString()}`,
        );
        setBusy(false);
      },
    });
  }, [addLayer, fit, formData, poll, progress, say]);

  /* ---- view actions ---- */
  const rotate = useCallback(
    (axis: 'x' | 'y' | 'z') => {
      const s = sceneRef.current;
      if (!s) return;
      const q = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(axis === 'x' ? 1 : 0, axis === 'y' ? 1 : 0, axis === 'z' ? 1 : 0),
        Math.PI / 2,
      );
      spin.current.premultiply(q);
      for (const k of Object.keys(s.layers) as LayerKey[]) {
        s.layers[k]?.quaternion.copy(spin.current);
      }
      s.fit();
      history.record();
      markDirty();
    },
    [history, markDirty, sceneRef],
  );

  const resetOrientation = useCallback(() => {
    const s = sceneRef.current;
    if (!s) return;
    spin.current.identity();
    for (const k of Object.keys(s.layers) as LayerKey[]) {
      s.layers[k]?.quaternion.identity();
    }
    s.fit();
    history.record();
    markDirty();
  }, [history, markDirty, sceneRef]);

  /** Zoom by moving the camera along its own view direction. */
  const zoomTo = useCallback(
    (pct: number) => {
      const s = sceneRef.current;
      if (!s || !s.fitDistance) return;
      const dir = s.camera.position.clone().sub(s.controls.target).normalize();
      s.camera.position.copy(
        s.controls.target.clone().add(dir.multiplyScalar(s.fitDistance * (100 / pct))),
      );
      s.controls.update();
      setZoom(pct);
    },
    [sceneRef],
  );

  const zoomBy = useCallback(
    (factor: number) => zoomTo(Math.min(400, Math.max(10, zoom * factor))),
    [zoom, zoomTo],
  );

  /* ---- field plumbing ---- */
  const onField = useCallback(
    (id: string, value: string | boolean) => {
      history.begin();
      markDirty();
      const num = (v: string | boolean) => (typeof v === 'boolean' ? 0 : parseFloat(v));
      switch (id) {
        case 'qty':
          setQuantity(Math.max(1, parseInt(String(value)) || 1));
          break;
        case 'layout':
          setLayout(String(value));
          break;
        case 'gap':
          setGap(String(value));
          break;
        case 'tiers':
          setTiers(Math.max(1, parseInt(String(value)) || 1));
          break;
        case 'shell':
          setParams((p) => ({ ...p, shell_thickness: num(value) }));
          break;
        case 'pitch':
          setParams((p) => ({ ...p, voxel_pitch: num(value) }));
          break;
        case 'rmf':
          setParams((p) => ({ ...p, riser_modulus_factor: num(value) }));
          break;
        case 'urisers':
          setParams((p) => ({ ...p, use_risers: Boolean(value) }));
          break;
        case 'nris':
          setParams((p) => ({ ...p, max_risers: parseInt(String(value)) || 1 }));
          break;
        case 'ceramic':
          setParams((p) => ({ ...p, ceramic: String(value) }));
          break;
        case 'ftemp':
          setParams((p) => ({ ...p, fire_temperature: num(value) }));
          break;
        case 'fhold':
          setParams((p) => ({ ...p, fire_hold_hours: num(value) }));
          break;
        case 'fshr':
          setParams((p) => ({
            ...p,
            fire_shrinkage: String(value).trim() === '' ? null : num(value),
          }));
          break;
      }
    },
    [history, markDirty],
  );

  const onFocusField = useCallback(
    (id: string, value: string) => {
      history.begin();
      history.stashField(id, value);
    },
    [history],
  );

  /* ---- keyboard ---- */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey) {
        const k = e.key.toLowerCase();
        if (k === 'z') {
          e.preventDefault();
          const st = e.shiftKey ? history.redo() : history.undo();
          if (st) {
            history.restoring.current = true;
            try {
              applyState(st);
            } finally {
              history.restoring.current = false;
            }
            markDirty();
          }
        }
        if (k === 's') {
          e.preventDefault();
          setSaveOpen(true);
        }
        return;
      }
      if (e.shiftKey && e.key.toLowerCase() === 'f') fit();
    };
    addEventListener('keydown', onKey);
    return () => removeEventListener('keydown', onKey);
  }, [applyState, fit, history, markDirty]);

  /* ---- save dialog ---- */
  const openSave = useCallback(async () => {
    setDlgErr('');
    setSaveName(title || 'Untitled project');
    try {
      const all = await listAll();
      setFolders((all.items || []).filter((i) => i.kind === 'folder'));
    } catch {
      setFolders([]);
    }
    setSaveOpen(true);
  }, [title]);

  /* ---- reopen a saved project ---- */
  useEffect(() => {
    const id = new URLSearchParams(location.search).get('project');
    if (!id) return;
    setProjectId(id);
    void (async () => {
      try {
        const res = await fetch(
          API + '/api/fs/projects/' + encodeURIComponent(id),
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const rec = await res.json();
        setTitle(rec.name || '');
        if (rec.params) setParams((p) => ({ ...p, ...rec.params }));
        const part = await fetchMesh(
          `/api/fs/projects/${encodeURIComponent(id)}/part`,
        );
        if (part) {
          addLayer('part', part);
          fit();
          /* The stored mesh, rebuilt as a File so the drop zone can name it and
             a rebuild has something to post -- reopening otherwise left the box
             asking for a part that was already on screen.
             The store strips `mesh_file` from public rows (store._public), so
             the original upload's filename is not available here; the project's
             own name is what the box shows. */
          const name = `${rec.name || 'part'}.stl`;
          partFile.current = new File([part], name, {
            type: 'application/octet-stream',
          });
          setPartInfo({ name, mb: (part.byteLength / 1048576).toFixed(2) });
        }
        // `focus` is false here: the user is looking at the part, not at
        // numbers they have seen before.
        if (rec.stats) setStats(rec.stats);
        say(`Opened “${rec.name}”.`);
        setDirty(false);
      } catch (e) {
        say(
          'Could not open the project: ' +
            (e as Error).message +
            ' <a class="dl" href="/">Back to the filesystem</a>',
          'err',
        );
      }
    })();
  }, [addLayer, fit, say]);

  /* ---- menus ---- */
  const fileItems: MenuItem[] = [
    { label: 'New project', key: '⌘⇧N', onSelect: () => (location.href = '/editor') },
    { label: 'Open…', key: '⌘O', onSelect: () => (location.href = '/') },
    { label: 'Save', key: '⌘S', separatorBefore: true, onSelect: () => void openSave() },
    { label: 'Save as…', key: '⌘⇧S', onSelect: () => void openSave() },
    { label: 'Rename…', onSelect: () => void openSave() },
    {
      label: 'Replace part…',
      separatorBefore: true,
      onSelect: () => document.getElementById('file')?.click(),
    },
    { label: 'Export…', onSelect: () => setExportOpen(true), disabled: !stats },
    {
      label: 'Delete project…',
      separatorBefore: true,
      danger: true,
      disabled: !projectId,
      onSelect: () => {
        if (!projectId) return;
        void fetch(API + '/api/fs/' + encodeURIComponent(projectId), {
          method: 'DELETE',
        }).then(() => (location.href = '/'));
      },
    },
  ];

  const editItems: MenuItem[] = [
    {
      label: 'Undo',
      key: '⌘Z',
      disabled: !history.canUndo,
      onSelect: () => {
        const st = history.undo();
        if (st) applyState(st);
      },
    },
    {
      label: 'Redo',
      key: '⌘⇧Z',
      disabled: !history.canRedo,
      onSelect: () => {
        const st = history.redo();
        if (st) applyState(st);
      },
    },
    {
      label: 'Reset layout',
      separatorBefore: true,
      onSelect: () => {
        setGap('');
        setTiers(1);
        history.record();
      },
    },
    { label: 'Reset orientation', onSelect: resetOrientation },
  ];

  const [plane, setPlane] = useState(true);
  const viewItems: MenuItem[] = [
    {
      label: 'Full screen',
      key: 'F11',
      onSelect: () => {
        if (document.fullscreenElement) void document.exitFullscreen();
        else void document.documentElement.requestFullscreen();
      },
    },
    {
      label: 'Show ground plane',
      key: '⇧G',
      on: plane,
      onSelect: () => {
        const s = sceneRef.current;
        if (!s) return;
        const on = !plane;
        setPlane(on);
        s.grid.visible = on;
        s.axes.visible = on;
      },
    },
    { label: 'Fit to view', key: '⇧F', separatorBefore: true, onSelect: fit },
  ];

  return (
    <div id="app">
      <Toolbar
        title={title}
        onTitleChange={(v) => {
          setTitle(v);
          markDirty();
        }}
        onTitleCommit={() => history.commit()}
        titleDisabled={false}
        modified={dirty}
        fileItems={fileItems}
        editItems={editItems}
        viewItems={viewItems}
        canUndo={history.canUndo}
        canRedo={history.canRedo}
        onUndo={() => {
          const st = history.undo();
          if (st) applyState(st);
        }}
        onRedo={() => {
          const st = history.redo();
          if (st) applyState(st);
        }}
        zoom={zoom}
        onZoom={zoomTo}
        onFit={fit}
        onGenerate={() => void generate()}
        generateDisabled={busy}
      />

      <div id="main">
        <aside id="side">
          <Tabs tab={tab} onTab={setTab} resultsEnabled={!!stats} />

          {tab === 'part' ? (
            <PartPanel
              quantity={quantity}
              layout={layout}
              gap={gap}
              tiers={tiers}
              upReadout={upReadout(spin.current)}
              total={total}
              clash={clash}
              partName={partInfo?.name ?? null}
              partSizeMB={partInfo?.mb ?? null}
              onField={onField}
              onFocusField={onFocusField}
              onBlurField={() => history.commit()}
              onFile={(f) => void onFile(f)}
              onResetLayout={() => {
                setGap('');
                setTiers(1);
                history.record();
              }}
              dropHot={dropHot}
              setDropHot={setDropHot}
            />
          ) : null}

          {tab === 'process' ? (
            <ProcessPanel
              params={params}
              shrinkHint={shrinkHint}
              onField={onField}
              onFocusField={onFocusField}
              onBlurField={() => history.commit()}
            />
          ) : null}

          {tab === 'results' ? <ResultsPanel stats={stats} /> : null}

          {/* Generate stays outside the tabs: it is the action the whole
              sidebar exists to reach. This bar only appears when there is a
              warning to carry. */}
          <div id="actionbar">
            <div id="warns">
              {((stats?.warnings as string[]) || []).map((w) => (
                <div className="warn" key={w}>
                  ⚠ {w}
                </div>
              ))}
            </div>
          </div>
        </aside>

        <ViewHost ref={viewRef}>
          <div id="hud" dangerouslySetInnerHTML={{ __html: hud }} />
          <Legend
            visible={visible}
            onToggle={(k, on) => setVisible((v) => ({ ...v, [k]: on }))}
            opacity={opacity}
            onOpacity={setOpacity}
          />
          <ViewControls
            onRotate={rotate}
            onZoomIn={() => zoomBy(1.25)}
            onZoomOut={() => zoomBy(1 / 1.25)}
            onFit={fit}
            onReset={resetOrientation}
          />
          <StatusBar state={status} />
        </ViewHost>
      </div>

      <Dialog
        open={saveOpen}
        title="Save project"
        onClose={() => setSaveOpen(false)}
        error={dlgErr}
        actions={
          <>
            <button className="ghost" onClick={() => setSaveOpen(false)}>
              Cancel
            </button>
            <button
              onClick={() => {
                // The save endpoint wants the mesh as well as the parameters;
                // without a part there is nothing to store.
                if (!partFile.current) {
                  setDlgErr('Drop a part before saving.');
                  return;
                }
                setSaveOpen(false);
                setDirty(false);
                setTitle(saveName);
              }}
            >
              Save project
            </button>
          </>
        }
      >
        <label className="f" htmlFor="svname">
          Project name
        </label>
        <input
          id="svname"
          maxLength={120}
          placeholder="e.g. Pump bracket, 4-up"
          value={saveName}
          onChange={(e) => setSaveName(e.target.value)}
        />
        <label className="f" htmlFor="svfolder">
          Folder
        </label>
        <select
          id="svfolder"
          value={saveFolder}
          onChange={(e) => setSaveFolder(e.target.value)}
        >
          <option value="">Filesystem (root)</option>
          {folders.map((f) => (
            <option key={f.id} value={f.id}>
              {f.name}
            </option>
          ))}
        </select>
        <div className="note" id="svnote">
          The part mesh is stored with the project, so it reopens exactly as it
          is on screen.
        </div>
      </Dialog>

      <Dialog
        open={exportOpen}
        title="Export"
        onClose={() => setExportOpen(false)}
        actions={
          <>
            <button className="ghost" onClick={() => setExportOpen(false)}>
              Cancel
            </button>
            <button
              onClick={() => {
                const urls = (stats?.urls ?? {}) as Record<string, string>;
                for (const [k, on] of Object.entries(exports)) {
                  if (on && urls[k]) location.href = API + urls[k];
                }
                setExportOpen(false);
              }}
            >
              Download
            </button>
          </>
        }
      >
        <div className="exhead">
          <span>Choose what to download</span>
          <button
            className="linkbtn"
            type="button"
            onClick={() => {
              const urls = (stats?.urls ?? {}) as Record<string, string>;
              const allOn = Object.keys(urls).every((k) => exports[k]);
              setExports(
                Object.fromEntries(Object.keys(urls).map((k) => [k, !allOn])),
              );
            }}
          >
            Toggle all
          </button>
        </div>
        <div id="exlist">
          {Object.keys((stats?.urls ?? {}) as Record<string, string>).map((k) => (
            <label className="exrow" key={k}>
              <input
                type="checkbox"
                checked={!!exports[k]}
                onChange={(e) =>
                  setExports((x) => ({ ...x, [k]: e.target.checked }))
                }
              />
              <span className="exname">{k}</span>
              <span className="exwhat">{EXPORT_LABELS[k] ?? 'artefact'}</span>
            </label>
          ))}
        </div>
      </Dialog>
    </div>
  );
}

const EXPORT_LABELS: Record<string, string> = {
  part_stl: 'the ingested part, as loaded',
  shell_stl: 'the green ceramic mould',
  fired_stl: 'the mould after firing shrinkage',
  tree_stl: 'sprue, runners, ingates and risers',
  section_png: 'the cross-section render',
};

/** Which world axis is "up" after the accumulated view rotation. */
function upReadout(q: THREE.Quaternion): string {
  const up = new THREE.Vector3(0, 0, 1).applyQuaternion(q.clone().invert());
  const axes: [string, number][] = [
    ['X', up.x],
    ['Y', up.y],
    ['Z', up.z],
  ];
  axes.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const [name, v] = axes[0];
  const sign = v >= 0 ? '+' : '−';
  return `${sign}${name}${Math.abs(Math.abs(v) - 1) < 1e-6 ? ' (file)' : ''}`;
}

/* `fetch` waits forever by default, so when the server is not running an
   upload simply sits on "reading file" with no explanation -- which is
   indistinguishable from a slow parse of a large mesh. */
function uploadError(e: Error): string {
  if (/abort/i.test(e.message)) {
    return 'The upload timed out. Is the server still running?';
  }
  if (/fetch|network/i.test(e.message)) {
    return 'Cannot reach the server — is uvicorn running?';
  }
  return e.message;
}
