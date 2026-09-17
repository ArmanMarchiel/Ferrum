import { useCallback, useEffect, useRef, useState } from 'react';
import { ItemIcon, FolderIcon } from '../components/Icons';
import { Dialog } from '../components/Dialog';
import { usePageClass } from '../hooks/usePageClass';
import {
  createFolder,
  deleteItem,
  listAll,
  listFolder,
  moveItem,
  renameItem,
  when,
} from '../services/api';
import type { Crumb, FsItem } from '../types/api';

/** Which dialog is open, and what it is acting on. */
type Modal =
  | { kind: 'none' }
  | { kind: 'newFolder' }
  | { kind: 'rename'; item: FsItem }
  | { kind: 'delete'; item: FsItem };

interface CtxState {
  item: FsItem;
  x: number;
  y: number;
}

declare global {
  interface Window {
    /** Test hook: the headless UI tests assert on the listing, not pixels. */
    __fs?: () => { parentId: string | null; crumbs: Crumb[]; items: FsItem[] };
  }
}

export function Home() {
  usePageClass('page-home');

  const [parentId, setParentId] = useState<string | null>(null);
  const [crumbs, setCrumbs] = useState<Crumb[]>([]);
  const [items, setItems] = useState<FsItem[]>([]);
  const [allItems, setAllItems] = useState<FsItem[]>([]);
  const [banner, setBanner] = useState('');
  const [modal, setModal] = useState<Modal>({ kind: 'none' });
  const [ctx, setCtx] = useState<CtxState | null>(null);
  const [newMenu, setNewMenu] = useState(false);

  // The row being dragged. A ref, not state: it changes during a drag and
  // re-rendering the list mid-drag cancels the drag in Safari.
  const dragging = useRef<FsItem | null>(null);

  /* ---------- listing ---------- */

  const load = useCallback(async (id: string | null) => {
    try {
      const d = await listFolder(id);
      setParentId(d.parent_id ?? null);
      setCrumbs(d.breadcrumbs || []);
      setItems(d.items || []);
      setBanner('');
    } catch (e) {
      const msg = (e as Error).message;
      // A folder deleted in another tab must not strand the page inside it.
      if (id && /no such item/.test(msg)) return load(null);
      setBanner(msg);
      setItems([]);
    }
    // Search degrades to empty; the listing has already reported any failure.
    try {
      setAllItems((await listAll()).items || []);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    void load(null);
  }, [load]);

  // Expose the same test hook the original page did.
  useEffect(() => {
    window.__fs = () => ({ parentId, crumbs, items });
    return () => {
      delete window.__fs;
    };
  }, [parentId, crumbs, items]);

  /* ---------- navigation ---------- */

  const open = useCallback(
    (it: FsItem) => {
      if (it.kind === 'folder') void load(it.id);
      else location.href = '/editor?project=' + encodeURIComponent(it.id);
    },
    [load],
  );

  const moveInto = useCallback(
    async (destId: string | null) => {
      const item = dragging.current;
      dragging.current = null;
      if (!item || item.parent_id === destId) return;
      try {
        await moveItem(item.id, destId);
        await load(parentId);
      } catch (e) {
        setBanner((e as Error).message);
      }
    },
    [load, parentId],
  );

  /* ---------- popovers ---------- */

  // One outside-click handler closes both popovers; each opener stops its own
  // click from reaching it.
  useEffect(() => {
    const onClick = () => {
      setNewMenu(false);
      setCtx(null);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      setNewMenu(false);
      setCtx(null);
      setModal({ kind: 'none' });
    };
    addEventListener('click', onClick);
    addEventListener('keydown', onKey);
    return () => {
      removeEventListener('click', onClick);
      removeEventListener('keydown', onKey);
    };
  }, []);

  return (
    <div id="app">
      <header id="brand">
        <h1>Ferrum</h1>
        <span className="sub">investment shell moulds</span>
      </header>

      <div id="pathbar">
        <Breadcrumbs
          crumbs={crumbs}
          onGo={(id) => void load(id)}
          onDrop={(id) => void moveInto(id)}
          dragging={dragging}
        />

        <Search items={allItems} onPick={open} onOpenFolder={(id) => void load(id)} />

        <div id="actions">
          <div className="menuwrap">
            <button
              className="btn primary"
              id="newbtn"
              aria-haspopup="true"
              aria-expanded={newMenu}
              onClick={(e) => {
                e.stopPropagation();
                setNewMenu((v) => !v);
              }}
            >
              + New <span style={{ fontSize: 9, opacity: 0.85 }}>▾</span>
            </button>
            <div className="menu" id="newmenu" hidden={!newMenu}>
              <button
                className="item"
                onClick={() => {
                  setNewMenu(false);
                  setModal({ kind: 'newFolder' });
                }}
              >
                <FolderIcon />
                Folder
              </button>
              <button
                className="item"
                onClick={() => {
                  setNewMenu(false);
                  location.href =
                    '/editor' +
                    (parentId ? '?folder=' + encodeURIComponent(parentId) : '');
                }}
              >
                <ItemIcon kind="project" />
                Part Mould
              </button>
            </div>
          </div>
        </div>
      </div>

      <div id="banner" hidden={!banner}>
        {banner}
      </div>

      <main id="listarea">
        <div id="card">
          <div className="head">
            <span>Name</span>
            <span>Type</span>
            <span>Updated</span>
          </div>
          <div id="rows">
            {items.length === 0 ? (
              <EmptyState inFolder={!!parentId} />
            ) : (
              items.map((it) => (
                <Row
                  key={it.id}
                  item={it}
                  onOpen={open}
                  onContext={(e) => {
                    e.preventDefault();
                    setCtx({ item: it, x: e.clientX, y: e.clientY });
                  }}
                  dragging={dragging}
                  onDropInto={(id) => void moveInto(id)}
                />
              ))
            )}
          </div>
        </div>
      </main>

      {ctx ? (
        <ContextMenu
          state={ctx}
          canMoveUp={!!parentId}
          onOpen={() => open(ctx.item)}
          onRename={() => setModal({ kind: 'rename', item: ctx.item })}
          onDelete={() => setModal({ kind: 'delete', item: ctx.item })}
          onMoveUp={() => {
            dragging.current = ctx.item;
            void moveInto(crumbs.length > 1 ? crumbs[crumbs.length - 2].id : null);
          }}
          onClose={() => setCtx(null)}
        />
      ) : null}

      <Modals
        modal={modal}
        parentId={parentId}
        onClose={() => setModal({ kind: 'none' })}
        onDone={() => {
          setModal({ kind: 'none' });
          void load(parentId);
        }}
      />
    </div>
  );
}

/* ---------- breadcrumbs ---------- */

function Breadcrumbs({
  crumbs,
  onGo,
  onDrop,
  dragging,
}: {
  crumbs: Crumb[];
  onGo: (id: string | null) => void;
  onDrop: (id: string | null) => void;
  dragging: React.MutableRefObject<FsItem | null>;
}) {
  const trail: Crumb[] = [{ id: null, name: 'Filesystem' }, ...crumbs];
  return (
    <nav id="crumbs" aria-label="Breadcrumb">
      {trail.map((c, i) => {
        const last = i === trail.length - 1;
        return (
          <span key={c.id ?? 'root'} style={{ display: 'contents' }}>
            {i ? <span className="sep">›</span> : null}
            {last ? (
              <span className="crumb current">{c.name}</span>
            ) : (
              <button
                className="crumb link"
                onClick={() => onGo(c.id)}
                /* A breadcrumb is a drop target too: it is the only way to
                   move something back out of the folder you are standing in. */
                onDragOver={(e) => {
                  if (dragging.current) {
                    e.preventDefault();
                    e.currentTarget.classList.add('drag-over');
                  }
                }}
                onDragLeave={(e) => e.currentTarget.classList.remove('drag-over')}
                onDrop={(e) => {
                  e.preventDefault();
                  e.currentTarget.classList.remove('drag-over');
                  onDrop(c.id);
                }}
              >
                {c.name}
              </button>
            )}
          </span>
        );
      })}
    </nav>
  );
}

/* ---------- rows ---------- */

function Row({
  item,
  onOpen,
  onContext,
  dragging,
  onDropInto,
}: {
  item: FsItem;
  onOpen: (it: FsItem) => void;
  onContext: (e: React.MouseEvent) => void;
  dragging: React.MutableRefObject<FsItem | null>;
  onDropInto: (id: string) => void;
}) {
  const isFolder = item.kind === 'folder';
  return (
    <div
      className="rowitem"
      draggable
      title={item.name}
      onClick={() => onOpen(item)}
      onContextMenu={onContext}
      onDragStart={(e) => {
        dragging.current = item;
        e.currentTarget.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        // Firefox refuses to start a drag without payload on the transfer.
        e.dataTransfer.setData('text/plain', item.id);
      }}
      onDragEnd={(e) => {
        dragging.current = null;
        e.currentTarget.classList.remove('dragging');
      }}
      onDragOver={
        isFolder
          ? (e) => {
              // Dropping a folder on itself is a no-op, not a move.
              if (!dragging.current || dragging.current.id === item.id) return;
              e.preventDefault();
              e.currentTarget.classList.add('drag-over');
            }
          : undefined
      }
      onDragLeave={
        isFolder ? (e) => e.currentTarget.classList.remove('drag-over') : undefined
      }
      onDrop={
        isFolder
          ? (e) => {
              e.preventDefault();
              e.currentTarget.classList.remove('drag-over');
              onDropInto(item.id);
            }
          : undefined
      }
    >
      <span className="nm">
        <ItemIcon kind={item.kind} />
        <span>{item.name}</span>
      </span>
      <span className="muted">{isFolder ? 'Folder' : 'Project'}</span>
      <span className="muted">{when(item.updated_at)}</span>
    </div>
  );
}

function EmptyState({ inFolder }: { inFolder: boolean }) {
  return (
    <div className="empty">
      <span style={{ width: 38, height: 38, opacity: 0.3 }}>
        <FolderIcon />
      </span>
      <p>{inFolder ? 'This folder is empty.' : 'No saved projects yet.'}</p>
      <div className="hint">
        A project is one editor session — the part, the process parameters and
        the arrangement on the plate. Open the editor, set it up, then press{' '}
        <b>Save</b>.
      </div>
      <button className="btn primary" onClick={() => (location.href = '/editor')}>
        Open the editor
      </button>
    </div>
  );
}

/* ---------- search ---------- */

function Search({
  items,
  onPick,
  onOpenFolder,
}: {
  items: FsItem[];
  onPick: (it: FsItem) => void;
  onOpenFolder: (id: string) => void;
}) {
  const [q, setQ] = useState('');
  const [cursor, setCursor] = useState(-1);
  const wrap = useRef<HTMLDivElement>(null);

  const hits = q.trim()
    ? items
        .filter((i) => i.name.toLowerCase().includes(q.trim().toLowerCase()))
        .slice(0, 40)
    : [];

  useEffect(() => setCursor(hits.length ? 0 : -1), [q, hits.length]);

  /* Opening a hit has to land the user somewhere they can see it, so a folder
     opens itself and a project opens in the editor. */
  const pick = (hit?: FsItem) => {
    if (!hit) return;
    setQ('');
    if (hit.kind === 'folder') onOpenFolder(hit.id);
    else onPick(hit);
  };

  return (
    <div id="searchwrap" ref={wrap} onClick={(e) => e.stopPropagation()}>
      <input
        id="search"
        type="search"
        autoComplete="off"
        spellCheck={false}
        placeholder="Search projects and folders…"
        aria-label="Search the filesystem"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            setQ('');
            return;
          }
          if (!hits.length) return;
          if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            setCursor(
              (c) =>
                (c + (e.key === 'ArrowDown' ? 1 : hits.length - 1)) % hits.length,
            );
          }
          if (e.key === 'Enter') {
            e.preventDefault();
            pick(hits[cursor]);
          }
        }}
      />
      <div id="results" role="listbox" hidden={!q.trim()}>
        {hits.length ? (
          hits.map((h, i) => (
            <button
              key={h.id}
              className={'hit' + (i === cursor ? ' on' : '')}
              onClick={() => pick(h)}
            >
              <span className="t">
                <ItemIcon kind={h.kind} />
                <span>{h.name}</span>
              </span>
              <span className="p">{h.kind === 'folder' ? 'Folder' : 'Project'}</span>
            </button>
          ))
        ) : (
          <div className="nohit">Nothing matches “{q.trim()}”.</div>
        )}
      </div>
    </div>
  );
}

/* ---------- context menu ---------- */

function ContextMenu({
  state,
  canMoveUp,
  onOpen,
  onRename,
  onDelete,
  onMoveUp,
  onClose,
}: {
  state: CtxState;
  canMoveUp: boolean;
  onOpen: () => void;
  onRename: () => void;
  onDelete: () => void;
  onMoveUp: () => void;
  onClose: () => void;
}) {
  const el = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: state.x, top: state.y });

  // Place it inside the viewport: near the cursor, but flipped when the menu
  // would otherwise run off the right or bottom edge.
  useEffect(() => {
    const r = el.current?.getBoundingClientRect();
    if (!r) return;
    setPos({
      left: Math.min(state.x, innerWidth - r.width - 8),
      top: Math.min(state.y, innerHeight - r.height - 8),
    });
  }, [state.x, state.y]);

  const run = (fn: () => void) => () => {
    onClose();
    fn();
  };

  return (
    <div id="ctx" ref={el} style={pos} onClick={(e) => e.stopPropagation()}>
      <button className="item" onClick={run(onOpen)}>
        {state.item.kind === 'folder' ? 'Open' : 'Open in editor'}
      </button>
      <button className="item" onClick={run(onRename)}>
        Rename…
      </button>
      {canMoveUp ? (
        <button className="item" onClick={run(onMoveUp)}>
          Move to parent folder
        </button>
      ) : null}
      <div className="msep" />
      <button className="item danger" onClick={run(onDelete)}>
        Delete…
      </button>
    </div>
  );
}

/* ---------- dialogs ---------- */

function Modals({
  modal,
  parentId,
  onClose,
  onDone,
}: {
  modal: Modal;
  parentId: string | null;
  onClose: () => void;
  onDone: () => void;
}) {
  const [name, setName] = useState('');
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);

  // Seed the field each time a dialog opens.
  useEffect(() => {
    setErr('');
    setBusy(false);
    setName(modal.kind === 'rename' ? modal.item.name : '');
  }, [modal]);

  if (modal.kind === 'none') return null;

  const submit = async () => {
    setBusy(true);
    setErr('');
    try {
      if (modal.kind === 'newFolder') {
        const n = name.trim();
        if (!n) throw new Error('Give the folder a name.');
        await createFolder(n, parentId);
      } else if (modal.kind === 'rename') {
        const n = name.trim();
        if (!n) throw new Error('The name cannot be empty.');
        if (n !== modal.item.name) await renameItem(modal.item.id, n);
      } else {
        await deleteItem(modal.item.id);
      }
      onDone();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const isDelete = modal.kind === 'delete';
  const title =
    modal.kind === 'newFolder'
      ? 'New folder'
      : modal.kind === 'rename'
        ? `Rename ${modal.item.kind}`
        : 'Delete';
  const ok =
    modal.kind === 'newFolder' ? 'Create folder' : isDelete ? 'Delete' : 'Rename';

  return (
    <Dialog
      open
      title={title}
      onClose={onClose}
      error={err}
      actions={
        <>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button
            className={'btn ' + (isDelete ? 'danger' : 'primary')}
            disabled={busy}
            onClick={() => void submit()}
          >
            {busy ? 'Working…' : ok}
          </button>
        </>
      }
    >
      {isDelete ? (
        /* Deleting a folder takes its contents with it, so say so before it
           happens rather than after -- there is no undo behind this. */
        <div className="body">
          {modal.item.kind === 'folder' ? (
            <>
              Delete the folder <b>{modal.item.name}</b> and{' '}
              <b>everything inside it</b>? Saved projects in this folder, and
              their part meshes, are removed for good.
            </>
          ) : (
            <>
              Delete the project <b>{modal.item.name}</b>? Its saved parameters
              and its part mesh are removed for good.
            </>
          )}
        </div>
      ) : (
        <>
          <label className="fieldlabel">
            {modal.kind === 'newFolder' ? 'Folder name' : 'Name'}
          </label>
          <input
            className="field"
            value={name}
            maxLength={120}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void submit();
            }}
          />
        </>
      )}
    </Dialog>
  );
}
