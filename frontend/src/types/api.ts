/* The backend's wire contract, in one place.
 *
 * These mirror what `backend/src/server/app.py` and `store.py` actually return.
 * They are hand-written rather than generated: the API is small and stable, and
 * a generator would pull in a build step for the sake of nine interfaces.
 *
 * When a route's shape changes, change it here first -- every caller is typed
 * against these, so the compiler will point at the code that needs updating.
 */

/** A row in the project store: either a folder or a saved project. */
export type ItemKind = 'folder' | 'project';

/** One entry in a folder listing (`store._public`). */
export interface FsItem {
  id: string;
  kind: ItemKind;
  name: string;
  parent_id: string | null;
  /** Unix seconds. Absent on rows that have never been written. */
  updated_at?: number;
  created_at?: number;
  /** Projects only: whether a part mesh has been stored. */
  has_mesh?: boolean;
  /** Projects only: which mould artefacts exist, e.g. ['shell', 'tree']. */
  mould?: string[];
}

/** One step of the breadcrumb trail, root excluded. */
export interface Crumb {
  id: string | null;
  name: string;
}

/** `GET /api/fs` — the contents of one folder. */
export interface FsListing {
  parent_id: string | null;
  breadcrumbs: Crumb[];
  items: FsItem[];
}

/** `GET /api/fs/all` — every row, flat, for the search index. */
export interface FsAll {
  items: FsItem[];
}

/** Where a job is in its lifecycle (`server.app` job registry). */
export type JobPhase = 'queued' | 'building' | 'done' | 'error';

/** The mesh and image URLs a finished job exposes. */
export interface JobUrls {
  part_stl?: string;
  shell_stl?: string;
  tree_stl?: string;
  shell_modified_stl?: string;
  section_png?: string;
  [key: string]: string | undefined;
}

/** Stats the pipeline reports back. Open-ended: the pipeline owns the keys. */
export interface JobStats {
  [key: string]: unknown;
}

/**
 * `POST /generate` and `GET /jobs/{id}/status`.
 *
 * Phase 1 returns immediately with the part and its URLs; `stats` is only
 * fully populated once `phase` is 'done'. See the two-phase design note in
 * the README -- the pipeline is CPU-bound and runs on a worker thread.
 */
export interface JobStatus {
  job_id: string;
  phase: JobPhase;
  urls: JobUrls;
  stats?: JobStats;
  /** Present when `phase` is 'error'. */
  error?: string;
  /** Free-text progress line shown under the status bar. */
  step?: string;
  /** 0..1, drives the progress bar. */
  progress?: number;
  warnings?: string[];
}

/** `GET /api/meta` — the alloys and defaults the sidebar is built from. */
export interface Meta {
  alloys: Record<string, AlloyMeta>;
  ceramics: Record<string, CeramicMeta>;
  defaults: PipelineParams;
  limits: { max_upload_mb: number };
  accepted_formats: string[];
}

export interface AlloyMeta {
  /** kg/mm^3 */
  density: number;
  label: string;
}

export interface CeramicMeta {
  label: string;
}

/**
 * The parameters the editor sends with a build. Names match the backend's
 * `Config` fields so the form can be posted without translation.
 *
 * Note `riser_modulus_factor`, not `riser_factor`: risers are sized by the
 * modulus method, and there is deliberately no `pour_time` or choke-area
 * parameter -- the Bernoulli fill-rate sizing was removed from `gating.py`
 * because the sprue doubles as a feeder and the modulus requirement always
 * won. See docs/pouring-funnels.md.
 */
export interface PipelineParams {
  alloy: string;
  shell_thickness: number;
  voxel_pitch: number;
  riser_modulus_factor: number;
  /** [sprue, runner, ingate] area ratio, e.g. [1, 2, 2]. */
  gating_ratio: number[];
  ceramic: string;
  /** degC */
  fire_temperature: number;
  fire_hold_hours: number;
  /** Linear fraction. null -> derived from the ceramic and firing schedule. */
  fire_shrinkage: number | null;
  use_risers: boolean;
  max_risers: number;
  quantity: number;
  layout: string;
  [key: string]: string | number | boolean | number[] | null | undefined;
}

/** A saved project as `GET /api/fs/projects/{id}` returns it. */
export interface Project extends FsItem {
  kind: 'project';
  params?: Partial<PipelineParams>;
  stats?: JobStats;
}

/** The four toggleable layers in the viewer. */
export type LayerKey = 'part' | 'shell' | 'fired' | 'tree';

/** Whether each layer is currently drawn. */
export type LayerVisibility = Record<LayerKey, boolean>;

/** The error shape both error paths use: FastAPI's `detail`, or our `error`. */
export interface ApiError {
  detail?: string;
  error?: string;
}
