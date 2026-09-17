import { useCallback, useRef } from 'react';
import { API } from '../services/api';
import type { JobStats, JobStatus } from '../types/api';

/* Polling a build to completion.
 *
 * The pipeline runs on a worker thread and takes seconds to minutes, so the
 * only way to find out where it is is to ask. The loop backs off from 350 ms
 * to 1.5 s so a long build is not a thousand requests.
 */

export interface Progress {
  label: string;
  frac: number;
  detail: string;
  secs: string;
}

export interface JobCallbacks {
  onProgress: (p: Progress) => void;
  /** Terminal failure. The message is ready to show. */
  onError: (msg: string) => void;
  /** The build finished; `stats` carries the URLs and the numbers. */
  onDone: (stats: JobStats, secs: string) => void | Promise<void>;
}

export function useJob() {
  /** The job currently being watched. A newer upload supersedes it. */
  const current = useRef<string | null>(null);

  const poll = useCallback(
    async (job: string, t0: number, cb: JobCallbacks) => {
      current.current = job;
      let delay = 350;
      let misses = 0; // consecutive "unknown job" replies

      for (;;) {
        await new Promise((r) => setTimeout(r, delay));
        delay = Math.min(delay * 1.2, 1500);
        if (current.current !== job) return; // superseded

        let res: Response;
        try {
          res = await fetch(`${API}/jobs/${job}/status`);
        } catch (e) {
          cb.onError('Connection lost while building: ' + (e as Error).message);
          return;
        }

        if (res.status === 404) {
          /* The server does not know this job. That means it restarted (its
             job registry is in memory), so the build will never report
             progress. Say so instead of spinning forever, which is what a
             490-second "building" message actually was. */
          if (++misses >= 3) {
            cb.onError(
              'The server restarted and lost this job. Reload the page (⌘⇧R) ' +
                'and upload again.',
            );
            return;
          }
          continue;
        }
        misses = 0;

        if (!res.ok) {
          cb.onError(`Lost the job on the server (HTTP ${res.status})`);
          return;
        }

        let st: JobStatus;
        try {
          st = (await res.json()) as JobStatus;
        } catch {
          cb.onError('Bad response from the server');
          return;
        }

        const secs = ((performance.now() - t0) / 1000).toFixed(0);

        if (st.phase === 'queued' || st.phase === 'building') {
          cb.onProgress({
            label: st.step || 'building mould',
            frac: st.progress ?? 0.1,
            detail: (st as { detail?: string }).detail || '',
            secs,
          });
          continue;
        }
        if (st.phase === 'error') {
          cb.onError('Pipeline error: ' + (st.error || 'unknown'));
          return;
        }
        if (st.phase === 'done') {
          cb.onProgress({
            label: 'loading meshes',
            frac: 0.97,
            detail: 'fetching shell and gating tree',
            secs,
          });
          await cb.onDone((st.stats ?? {}) as JobStats, secs);
          return;
        }
      }
    },
    [],
  );

  return { poll, current };
}
