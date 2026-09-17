import { useEffect, useRef, useState, type ReactNode } from 'react';

/* The editor's header.
 *
 * Two rows in the Google Docs manner: the mark on the left spanning both, the
 * document's name on top, and the menus beneath it. The name is the first
 * thing read and the menus hang off it, which is the whole reason that header
 * reads as calmly as it does.
 */

export interface MenuItem {
  label: string;
  /** The shortcut, ranged right. */
  key?: string;
  onSelect?: () => void;
  disabled?: boolean;
  danger?: boolean;
  /** A toggle that shows its state with a tick. */
  on?: boolean;
  /** A rule above this item. */
  separatorBefore?: boolean;
}

/** One menu on the bar: a trigger that reports its open state, plus items. */
export function Menu({
  id,
  label,
  items,
  open,
  onToggle,
  triggerClass = 'menubtn',
  menuId,
  children,
}: {
  id: string;
  label: ReactNode;
  items: MenuItem[];
  open: boolean;
  onToggle: () => void;
  triggerClass?: string;
  menuId?: string;
  children?: ReactNode;
}) {
  return (
    <div className="menuwrap">
      <button
        id={id}
        className={triggerClass}
        aria-haspopup="true"
        aria-expanded={open}
        onClick={(e) => {
          e.stopPropagation();
          onToggle();
        }}
      >
        {label}
        {children}
      </button>
      <div className="menu" id={menuId} hidden={!open} role="menu">
        {items.map((it, i) => (
          <span key={it.label + i} style={{ display: 'contents' }}>
            {it.separatorBefore ? <div className="msep" /> : null}
            <button
              className={
                'mi' + (it.danger ? ' danger' : '') + (it.on ? ' on' : '')
              }
              disabled={it.disabled}
              onClick={() => it.onSelect?.()}
            >
              {it.label}
              {it.key ? <span className="k">{it.key}</span> : null}
            </button>
          </span>
        ))}
      </div>
    </div>
  );
}

const UndoIcon = () => (
  <svg
    width="15"
    height="15"
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M3 8h7a3.2 3.2 0 0 1 0 6.4H6" />
    <path d="M5.6 5.4 3 8l2.6 2.6" />
  </svg>
);

const RedoIcon = () => (
  <svg
    width="15"
    height="15"
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M13 8H6a3.2 3.2 0 0 0 0 6.4h4" />
    <path d="M10.4 5.4 13 8l-2.6 2.6" />
  </svg>
);

export const ZOOM_STEPS = [50, 75, 90, 100, 125, 150, 200];

export interface ToolbarProps {
  title: string;
  onTitleChange: (v: string) => void;
  onTitleCommit: () => void;
  titleDisabled: boolean;
  modified: boolean;

  fileItems: MenuItem[];
  editItems: MenuItem[];
  viewItems: MenuItem[];

  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;

  zoom: number;
  onZoom: (pct: number) => void;
  onFit: () => void;

  onGenerate: () => void;
  generateDisabled: boolean;
}

export function Toolbar(props: ToolbarProps) {
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const bar = useRef<HTMLElement>(null);

  // One outside-click handler closes whichever menu is open; each trigger
  // stops its own click from reaching it.
  useEffect(() => {
    const close = () => setOpenMenu(null);
    addEventListener('click', close);
    return () => removeEventListener('click', close);
  }, []);

  const toggle = (name: string) =>
    setOpenMenu((cur) => (cur === name ? null : name));

  // Selecting an item closes the menu it came from.
  const wrap = (items: MenuItem[]): MenuItem[] =>
    items.map((it) => ({
      ...it,
      onSelect: () => {
        setOpenMenu(null);
        it.onSelect?.();
      },
    }));

  return (
    <header id="toolbar" ref={bar}>
      {/* The mark doubles as the way back to the filesystem, the way a Docs
          icon returns to the document list. */}
      <a
        id="logo"
        href="/"
        title="Back to the filesystem"
        aria-label="Back to the filesystem"
      >
        <img src="/ferrum-logo.png" alt="Ferrum" />
      </a>

      <div id="headrows">
        <span className="titlewrap">
          <input
            id="title"
            className={props.modified ? 'modified' : ''}
            placeholder="Untitled project"
            maxLength={120}
            disabled={props.titleDisabled}
            aria-label="Project name"
            spellCheck={false}
            value={props.title}
            onChange={(e) => props.onTitleChange(e.target.value)}
            onBlur={props.onTitleCommit}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur();
            }}
          />
          <span className="dot" aria-hidden="true" />
        </span>

        <nav id="menubar" aria-label="Menus">
          <Menu
            id="filebtn"
            menuId="filemenu"
            label="File"
            items={wrap(props.fileItems)}
            open={openMenu === 'file'}
            onToggle={() => toggle('file')}
          />
          <Menu
            id="editbtn"
            menuId="editmenu"
            label="Edit"
            items={wrap(props.editItems)}
            open={openMenu === 'edit'}
            onToggle={() => toggle('edit')}
          />
          <Menu
            id="viewbtn"
            menuId="viewmenu"
            label="View"
            items={wrap(props.viewItems)}
            open={openMenu === 'view'}
            onToggle={() => toggle('view')}
          />
        </nav>
      </div>

      <span className="gap" />

      {/* The actions, held together in the middle of the bar and kept clear of
          the text menus on the left: these are things done TO the model, not
          places to go. */}
      <div id="actions" role="group" aria-label="Actions">
        <div id="editbar" role="group" aria-label="Undo and redo">
          <button
            id="undo"
            className="iconbtn"
            disabled={!props.canUndo}
            title="Undo (⌘Z)"
            aria-label="Undo"
            onClick={props.onUndo}
          >
            <UndoIcon />
          </button>
          <button
            id="redo"
            className="iconbtn"
            disabled={!props.canRedo}
            title="Redo (⌘⇧Z)"
            aria-label="Redo"
            onClick={props.onRedo}
          >
            <RedoIcon />
          </button>
        </div>

        <Menu
          id="zoombtn"
          menuId="zoommenu"
          triggerClass="zoomtrigger"
          label={<span id="zoomval">{Math.round(props.zoom)}%</span>}
          open={openMenu === 'zoom'}
          onToggle={() => toggle('zoom')}
          items={wrap([
            { label: 'Fit', key: '⇧F', onSelect: props.onFit },
            ...ZOOM_STEPS.map((p, i) => ({
              label: `${p}%`,
              on: Math.round(props.zoom) === p,
              separatorBefore: i === 0,
              onSelect: () => props.onZoom(p),
            })),
          ])}
        >
          <span className="chev">▾</span>
        </Menu>

        <button id="go" disabled={props.generateDisabled} onClick={props.onGenerate}>
          Generate mould
        </button>
      </div>

      <span className="gap" />
    </header>
  );
}
