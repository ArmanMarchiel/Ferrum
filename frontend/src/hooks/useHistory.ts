import { useCallback, useRef, useState } from 'react';

/* Undo / redo.
 *
 * The editor's document is exactly what `capture()` returns -- the parameters,
 * the arrangement and the rotation -- so history is a stack of those snapshots
 * rather than a log of inverse operations. Snapshots are whole-state and small
 * (a few hundred bytes), and replaying one is just `apply`, which every other
 * restore path already uses.
 *
 * `undoStack` holds states already left behind, newest last; `redoStack` holds
 * those stepped back out of. The state on screen belongs to neither -- it is
 * read live -- which is what keeps the two stacks from double-counting it.
 */

const HISTORY_LIMIT = 60;

const equal = <T,>(a: T | null, b: T | null) =>
  !!a && !!b && JSON.stringify(a) === JSON.stringify(b);

export interface History<T> {
  canUndo: boolean;
  canRedo: boolean;
  /** True while a snapshot is being replayed; suppresses recording. */
  restoring: React.MutableRefObject<boolean>;
  /** Call before a change. Safe to call repeatedly during one edit. */
  begin: () => void;
  /** Call after a change has settled. */
  commit: () => void;
  /** One-shot edits (a gizmo drop, a 90° turn): both halves at once. */
  record: () => void;
  /** Put a field's pre-edit value back into the open base. */
  stashField: (id: string, value: unknown) => void;
  undo: () => T | null;
  redo: () => T | null;
  reset: () => void;
}

export function useHistory<T extends object>(
  capture: () => T,
): History<T> {
  const undoStack = useRef<T[]>([]);
  const redoStack = useRef<T[]>([]);
  const restoring = useRef(false);
  const pending = useRef<number | null>(null);

  /* The snapshot as it stood before the edit currently being made. Committing
     pushes THIS, not the state after it -- undo has to return to where the
     user was, and the state after is still on screen to be read. */
  const base = useRef<T | null>(null);

  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const sync = useCallback(() => {
    setCanUndo(undoStack.current.length > 0);
    setCanRedo(redoStack.current.length > 0);
  }, []);

  /* Typing in a field fires per keystroke, so the base is only taken once and
     held until the edit is committed -- otherwise "6" typed over "1" would
     leave two snapshots and take two undos to clear. */
  const begin = useCallback(() => {
    if (restoring.current) return;
    if (base.current === null) base.current = capture();
  }, [capture]);

  const commit = useCallback(() => {
    if (restoring.current || base.current === null) return;
    const before = base.current;
    base.current = null;
    // A field focused and left untouched must not fill the stack with states
    // identical to the one already on top.
    if (equal(before, capture())) return;
    undoStack.current.push(before);
    if (undoStack.current.length > HISTORY_LIMIT) undoStack.current.shift();
    redoStack.current.length = 0; // a new edit forks the timeline
    sync();
  }, [capture, sync]);

  const record = useCallback(() => {
    begin();
    if (pending.current !== null) clearTimeout(pending.current);
    pending.current = window.setTimeout(commit, 250);
  }, [begin, commit]);

  /* captureState reads the live controls, so once typing starts the "before"
     it would report is already the after; this puts the original back. */
  const stashField = useCallback((id: string, value: unknown) => {
    if (restoring.current || !base.current) return;
    const fields = (base.current as { fields?: Record<string, unknown> }).fields;
    if (fields && id in fields) fields[id] = value;
  }, []);

  const undo = useCallback((): T | null => {
    // Flush an edit still being typed, so undo steps back over it rather than
    // over the one before it.
    if (pending.current !== null) clearTimeout(pending.current);
    commit();
    if (!undoStack.current.length) return null;
    redoStack.current.push(capture());
    const st = undoStack.current.pop() ?? null;
    sync();
    return st;
  }, [capture, commit, sync]);

  const redo = useCallback((): T | null => {
    if (!redoStack.current.length) return null;
    undoStack.current.push(capture());
    const st = redoStack.current.pop() ?? null;
    sync();
    return st;
  }, [capture, sync]);

  const reset = useCallback(() => {
    undoStack.current = [];
    redoStack.current = [];
    base.current = null;
    sync();
  }, [sync]);

  return {
    canUndo,
    canRedo,
    restoring,
    begin,
    commit,
    record,
    stashField,
    undo,
    redo,
    reset,
  };
}
