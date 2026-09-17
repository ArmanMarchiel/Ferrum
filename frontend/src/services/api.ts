/* The one place the frontend talks to the backend.
 *
 * Every call goes through `api()` so a dead server or a rejected operation
 * surfaces as one readable Error rather than a silent no-op -- that was the
 * rule in the original home.html and it is worth keeping: the most common
 * failure in development is uvicorn not running, and a bare fetch rejection
 * says nothing useful about that.
 */
import type {
  FsAll,
  FsItem,
  FsListing,
  JobStatus,
  Meta,
  ApiError,
} from '../types/api';

/** Same origin in production; Vite proxies to uvicorn in dev. */
export const API = typeof location === 'undefined' ? '' : location.origin;

export async function api<T>(path: string, opts?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API + path, opts);
  } catch (e) {
    throw new Error(
      'Cannot reach the server — is uvicorn running? ' + (e as Error).message,
    );
  }
  const data = (await res.json().catch(() => ({}))) as T & ApiError;
  if (!res.ok) {
    // FastAPI raises with `detail`; the pipeline's own errors use `error`.
    throw new Error(data.detail || data.error || `HTTP ${res.status}`);
  }
  return data;
}

/* ---- filesystem ---- */

export const listFolder = (parent: string | null) =>
  api<FsListing>('/api/fs' + (parent ? `?parent=${encodeURIComponent(parent)}` : ''));

export const listAll = () => api<FsAll>('/api/fs/all');

export const createFolder = (name: string, parent: string | null) =>
  api<FsItem>('/api/fs/folders', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name, parent_id: parent }),
  });

export const patchItem = (id: string, body: Record<string, unknown>) =>
  api<FsItem>(`/api/fs/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });

export const renameItem = (id: string, name: string) => patchItem(id, { name });

export const moveItem = (id: string, parent_id: string | null) =>
  patchItem(id, { parent_id });

export const deleteItem = (id: string) =>
  api<{ ok: boolean }>(`/api/fs/${encodeURIComponent(id)}`, { method: 'DELETE' });

/* ---- projects and jobs ---- */

export const getMeta = () => api<Meta>('/api/meta');

export const getProject = (id: string) =>
  api<FsItem>(`/api/fs/projects/${encodeURIComponent(id)}`);

export const jobStatus = (job: string) =>
  api<JobStatus>(`/jobs/${encodeURIComponent(job)}/status`);

/** Fetch a mesh as raw bytes for the STL loader. Returns null when absent. */
export async function fetchMesh(url: string): Promise<ArrayBuffer | null> {
  const res = await fetch(API + url);
  return res.ok ? res.arrayBuffer() : null;
}

/** Relative time, matching the filesystem's original `when()`. */
export function when(t?: number): string {
  if (!t) return '—';
  const d = new Date(t * 1000);
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
    : d.toLocaleDateString([], {
        month: 'short',
        day: 'numeric',
        year: d.getFullYear() === now.getFullYear() ? undefined : 'numeric',
      });
}
