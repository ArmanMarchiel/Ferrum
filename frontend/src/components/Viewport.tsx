import { forwardRef } from 'react';
import type { LayerKey, LayerVisibility } from '../types/api';

/* Everything drawn over the 3-D canvas: the HUD, the legend, the view
   controls and the status line. The canvas itself is appended by useScene --
   React owns the chrome, three.js owns the pixels. */

const LAYER_ROWS: { key: LayerKey; label: string; swatch: string }[] = [
  { key: 'part', label: 'part', swatch: 'var(--part)' },
  { key: 'shell', label: 'shell', swatch: 'var(--shell)' },
  { key: 'fired', label: 'shell, fired', swatch: 'var(--fired)' },
  { key: 'tree', label: 'gating tree', swatch: 'var(--tree)' },
];

export function Legend({
  visible,
  onToggle,
  opacity,
  onOpacity,
}: {
  visible: LayerVisibility;
  onToggle: (k: LayerKey, on: boolean) => void;
  opacity: number;
  onOpacity: (v: number) => void;
}) {
  return (
    <div id="legend">
      {LAYER_ROWS.map((r) => (
        <label key={r.key} data-k={r.key}>
          <input
            type="checkbox"
            id={`c-${r.key}`}
            checked={visible[r.key]}
            onChange={(e) => onToggle(r.key, e.target.checked)}
          />
          <span className="sw" style={{ background: r.swatch }} />
          {r.label}
        </label>
      ))}
      <div className="opa">
        <span>opacity</span>
        <input
          id="opacity"
          type="range"
          min={5}
          max={100}
          value={opacity}
          onChange={(e) => onOpacity(Number(e.target.value))}
        />
      </div>
    </div>
  );
}

export function ViewControls({
  onRotate,
  onZoomIn,
  onZoomOut,
  onFit,
  onReset,
}: {
  onRotate: (axis: 'x' | 'y' | 'z') => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFit: () => void;
  onReset: () => void;
}) {
  return (
    <div id="viewctl">
      <span className="lbl">rotate</span>
      {(['x', 'y', 'z'] as const).map((a) => (
        <button
          key={a}
          data-rot={a}
          title={`turn 90° about ${a.toUpperCase()}`}
          onClick={() => onRotate(a)}
        >
          {a.toUpperCase()}
        </button>
      ))}
      <span className="sep" />
      <span className="lbl">zoom</span>
      <button id="zoom-out" title="zoom out" onClick={onZoomOut}>
        −
      </button>
      <button id="zoom-in" title="zoom in" onClick={onZoomIn}>
        +
      </button>
      <span className="sep" />
      <button id="fit" className="wide" title="frame everything" onClick={onFit}>
        Fit
      </button>
      <button id="reset" className="wide" title="clear rotation" onClick={onReset}>
        Reset
      </button>
    </div>
  );
}

export interface StatusState {
  /** HTML is allowed here: the original composed warning markup into it. */
  html: string;
  cls: '' | 'err' | 'warned' | 'busy';
  /** 0..1 when a build is running; null hides the bar. */
  frac: number | null;
  step?: string;
}

export function StatusBar({ state }: { state: StatusState }) {
  return (
    <div id="status" className={state.cls}>
      <span dangerouslySetInnerHTML={{ __html: state.html }} />
      {state.frac !== null ? (
        <>
          <div id="bar">
            <i style={{ width: `${Math.round(state.frac * 100)}%` }} />
          </div>
          {state.step ? <div className="step">{state.step}</div> : null}
        </>
      ) : null}
    </div>
  );
}

/** The host element the three.js canvas is appended into. */
export const ViewHost = forwardRef<HTMLElement, { children?: React.ReactNode }>(
  function ViewHost({ children }, ref) {
    return (
      <main id="view" ref={ref as React.Ref<HTMLElement>}>
        {children}
      </main>
    );
  },
);
