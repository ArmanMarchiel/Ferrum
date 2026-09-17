import type { ChangeEvent, ReactNode } from 'react';
import type { JobStats, PipelineParams } from '../types/api';

/* The sidebar: Part, Process and Results.
 *
 * The sidebar holds three panels' worth of controls, and stacking them all
 * vertically pushed Generate below the fold on a laptop. Generate now lives in
 * the toolbar, so it is reachable from any tab.
 */

export type TabName = 'part' | 'process' | 'results';

/** A labelled control on one row. */
function Row({
  label,
  htmlFor,
  children,
  hidden,
  id,
}: {
  label: string;
  htmlFor?: string;
  children: ReactNode;
  hidden?: boolean;
  id?: string;
}) {
  return (
    <div className="row" id={id} style={hidden ? { display: 'none' } : undefined}>
      <label htmlFor={htmlFor}>{label}</label>
      {children}
    </div>
  );
}

export interface FieldProps {
  id: string;
  value: number | string;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
  onFocus?: () => void;
  onBlur?: () => void;
  step?: number;
  min?: number;
  max?: number;
  placeholder?: string;
  type?: string;
}

const NumField = (p: FieldProps) => (
  <input
    id={p.id}
    type={p.type ?? 'number'}
    step={p.step}
    min={p.min}
    max={p.max}
    placeholder={p.placeholder}
    value={p.value}
    onChange={p.onChange}
    onFocus={p.onFocus}
    onBlur={p.onBlur}
  />
);

export function Tabs({
  tab,
  onTab,
  resultsEnabled,
}: {
  tab: TabName;
  onTab: (t: TabName) => void;
  resultsEnabled: boolean;
}) {
  const names: TabName[] = ['part', 'process', 'results'];
  return (
    <nav id="tabs" role="tablist">
      {names.map((n) => (
        <button
          key={n}
          className={'tab' + (tab === n ? ' on' : '')}
          data-tab={n}
          role="tab"
          disabled={n === 'results' && !resultsEnabled}
          onClick={() => onTab(n)}
        >
          {n[0].toUpperCase() + n.slice(1)}
        </button>
      ))}
    </nav>
  );
}

export interface PartPanelProps {
  quantity: number;
  layout: string;
  gap: string;
  tiers: number;
  upReadout: string;
  total: string;
  clash: string;
  /** The loaded part, so the drop zone can name it instead of prompting. */
  partName: string | null;
  partSizeMB: string | null;
  onField: (id: string, value: string) => void;
  onFocusField: (id: string, value: string) => void;
  onBlurField: () => void;
  onFile: (f: File) => void;
  onResetLayout: () => void;
  dropHot: boolean;
  setDropHot: (v: boolean) => void;
}

export function PartPanel(p: PartPanelProps) {
  const multi = p.quantity > 1;
  return (
    <div className="sec" data-panel="part">
      {/* A div, not a label: a label is display:inline, so the dashed box did
          not lay out as a block and its contents spilled out of it. The click
          is forwarded to the hidden input by hand instead. */}
      <div
        id="drop"
        className={
          (p.dropHot ? 'hot' : '') + (p.partName ? ' loaded' : '')
        }
        role="button"
        tabIndex={0}
        title={p.partName ? `${p.partName} — click to replace` : undefined}
        onClick={() => document.getElementById('file')?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            document.getElementById('file')?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          p.setDropHot(true);
        }}
        onDragLeave={() => p.setDropHot(false)}
        onDrop={(e) => {
          e.preventDefault();
          p.setDropHot(false);
          const f = e.dataTransfer.files?.[0];
          if (f) p.onFile(f);
        }}
      >
        {p.partName ? (
          /* Once a part is loaded the box names it, so the sidebar says which
             mesh is on screen rather than still asking for one. */
          <>
            <b className="fname">{p.partName}</b>
            <br />
            <span style={{ fontSize: 11 }}>{p.partSizeMB} MB</span>
          </>
        ) : (
          <>
            Drop an <b>STL</b> here
            <br />
            <span style={{ fontSize: 11 }}>also OBJ · PLY · OFF · 3MF</span>
          </>
        )}
      </div>
      <input
        type="file"
        id="file"
        accept=".stl,.obj,.ply,.off,.3mf"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) p.onFile(f);
          e.target.value = '';
        }}
      />

      <Row label="quantity (parts/mould)" htmlFor="qty">
        <NumField
          id="qty"
          step={1}
          min={1}
          value={p.quantity}
          onChange={(e) => p.onField('qty', e.target.value)}
          onFocus={() => p.onFocusField('qty', String(p.quantity))}
          onBlur={p.onBlurField}
        />
      </Row>

      <Row label="arrangement" htmlFor="layout" id="layoutrow" hidden={!multi}>
        <select
          id="layout"
          value={p.layout}
          onChange={(e) => p.onField('layout', e.target.value)}
        >
          <option value="grid">grid</option>
          <option value="radial">radial cluster</option>
        </select>
      </Row>

      <Row label="part gap (mm)" htmlFor="gap" id="gaprow" hidden={!multi}>
        <NumField
          id="gap"
          step={1}
          min={0}
          placeholder="auto"
          value={p.gap}
          onChange={(e) => p.onField('gap', e.target.value)}
          onFocus={() => p.onFocusField('gap', p.gap)}
          onBlur={p.onBlurField}
        />
      </Row>

      <Row label="stack tiers" htmlFor="tiers" id="tierrow" hidden={!multi}>
        <NumField
          id="tiers"
          step={1}
          min={1}
          value={p.tiers}
          onChange={(e) => p.onField('tiers', e.target.value)}
          onFocus={() => p.onFocusField('tiers', String(p.tiers))}
          onBlur={p.onBlurField}
        />
      </Row>

      <div id="arrangebar" style={{ display: multi ? 'block' : 'none', marginTop: 6 }}>
        <button id="resetlayout" className="mini" onClick={p.onResetLayout}>
          Reset layout
        </button>
        <div
          style={{
            fontSize: 10,
            color: 'var(--dim)',
            marginTop: 5,
            lineHeight: 1.5,
          }}
        >
          Click a part to select · drag arrows to move · Tab for rotate
          <br />
          Alt = move all · Shift = free angle
        </div>
      </div>

      <div className="grp" style={{ marginTop: 14 }}>
        pour direction
      </div>
      <Row label="up is">
        <span id="upreadout" style={{ fontSize: 11 }}>
          {p.upReadout}
        </span>
      </Row>
      <div className="hint">
        Up is the top of the ground plane. Turn the part with the <b>X / Y / Z</b>{' '}
        buttons until it stands the way it should in the furnace — the sprue
        rises out of whatever face is then pointing up. Orbiting only moves the
        camera and does not change this.
      </div>

      {p.total ? (
        <div id="total" style={{ fontSize: 11, color: 'var(--dim)', marginTop: 6 }}>
          {p.total}
        </div>
      ) : null}
      {p.clash ? <div id="clash">{p.clash}</div> : null}
    </div>
  );
}

export interface ProcessPanelProps {
  params: PipelineParams;
  shrinkHint: string;
  onField: (id: string, value: string | boolean) => void;
  onFocusField: (id: string, value: string) => void;
  onBlurField: () => void;
}

export function ProcessPanel(p: ProcessPanelProps) {
  const v = p.params;
  return (
    <div className="sec" data-panel="process">
      {/* Grouped by which shell each input actually changes. Every control here
          provably alters the output; parameters that did not (pour time,
          discharge coefficient, metallostatic head) have been removed rather
          than left on screen implying they do something. */}
      <div className="grp">green shell — as built</div>

      <Row label="shell thickness (mm)" htmlFor="shell">
        <NumField
          id="shell"
          step={0.5}
          min={0.5}
          value={v.shell_thickness}
          onChange={(e) => p.onField('shell', e.target.value)}
          onFocus={() => p.onFocusField('shell', String(v.shell_thickness))}
          onBlur={p.onBlurField}
        />
      </Row>
      <Row label="voxel pitch (mm)" htmlFor="pitch">
        <NumField
          id="pitch"
          step={0.25}
          min={0.25}
          value={v.voxel_pitch}
          onChange={(e) => p.onField('pitch', e.target.value)}
          onFocus={() => p.onFocusField('pitch', String(v.voxel_pitch))}
          onBlur={p.onBlurField}
        />
      </Row>
      <Row label="feeder factor" htmlFor="rmf">
        <NumField
          id="rmf"
          step={0.05}
          min={1}
          value={v.riser_modulus_factor}
          onChange={(e) => p.onField('rmf', e.target.value)}
          onFocus={() => p.onFocusField('rmf', String(v.riser_modulus_factor))}
          onBlur={p.onBlurField}
        />
      </Row>
      <Row label="separate risers" htmlFor="urisers">
        <input
          id="urisers"
          type="checkbox"
          checked={v.use_risers}
          onChange={(e) => p.onField('urisers', e.target.checked)}
        />
      </Row>
      <Row
        label="max risers per part"
        htmlFor="nris"
        id="nrisrow"
        hidden={!v.use_risers}
      >
        <NumField
          id="nris"
          step={1}
          min={1}
          value={v.max_risers}
          onChange={(e) => p.onField('nris', e.target.value)}
          onFocus={() => p.onFocusField('nris', String(v.max_risers))}
          onBlur={p.onBlurField}
        />
      </Row>
      <div className="hint">
        Feeder factor sets how much longer the sprue must stay molten than the
        section it feeds (1.2 = 20% margin). The sprue already feeds every
        thermal centre within reach, so separate risers are only needed for an
        isolated heavy section — leave them off unless a casting shows shrinkage
        away from the sprue.
      </div>

      <div className="grp">fired shell — after burnout</div>
      <Row label="ceramic system" htmlFor="ceramic">
        <select
          id="ceramic"
          value={v.ceramic}
          onChange={(e) => p.onField('ceramic', e.target.value)}
        >
          <option value="fused_silica">fused silica</option>
          <option value="zircon">zircon / fused silica</option>
          <option value="alumina">alumino-silicate</option>
        </select>
      </Row>
      <Row label="firing temp (°C)" htmlFor="ftemp">
        <NumField
          id="ftemp"
          step={25}
          min={600}
          max={1600}
          value={v.fire_temperature}
          onChange={(e) => p.onField('ftemp', e.target.value)}
          onFocus={() => p.onFocusField('ftemp', String(v.fire_temperature))}
          onBlur={p.onBlurField}
        />
      </Row>
      <Row label="hold time (h)" htmlFor="fhold">
        <NumField
          id="fhold"
          step={0.5}
          min={0.1}
          value={v.fire_hold_hours}
          onChange={(e) => p.onField('fhold', e.target.value)}
          onFocus={() => p.onFocusField('fhold', String(v.fire_hold_hours))}
          onBlur={p.onBlurField}
        />
      </Row>
      <Row label="shrinkage (%)" htmlFor="fshr">
        <NumField
          id="fshr"
          step={0.01}
          min={0}
          placeholder="auto"
          value={v.fire_shrinkage ?? ''}
          onChange={(e) => p.onField('fshr', e.target.value)}
          onFocus={() => p.onFocusField('fshr', String(v.fire_shrinkage ?? ''))}
          onBlur={p.onBlurField}
        />
      </Row>
      <div className="hint" id="shrinkhint">
        {p.shrinkHint}
      </div>
    </div>
  );
}

/** The stats table. The pipeline owns the keys, so rows are rendered as given. */
export function ResultsPanel({ stats }: { stats: JobStats | null }) {
  const rows = stats ? statRows(stats) : [];
  return (
    <div className="sec" data-panel="results">
      {rows.length ? (
        <table id="stats">
          <tbody>
            {rows.map(([k, v]) => (
              <tr key={k}>
                <td>{k}</td>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div id="noresults" style={{ fontSize: 11, color: 'var(--dim)' }}>
          No mould yet. Press <b>Generate mould</b>.
        </div>
      )}
    </div>
  );
}

/** Flatten the stats the pipeline reports into label/value pairs. */
function statRows(d: JobStats): [string, string][] {
  const out: [string, string][] = [];
  const num = (x: unknown, dp = 2) =>
    typeof x === 'number' ? x.toFixed(dp) : String(x ?? '—');

  const part = d.part as Record<string, unknown> | undefined;
  if (part) {
    out.push(['part volume (mm³)', num(part.volume, 0)]);
    out.push(['part area (mm²)', num(part.area, 0)]);
    if (part.modulus !== undefined) out.push(['modulus (mm)', num(part.modulus)]);
  }
  const shell = d.shell as Record<string, unknown> | undefined;
  if (shell) {
    out.push(['shell watertight', shell.watertight ? 'yes' : 'no']);
    const t = shell.thickness_measured as Record<string, unknown> | undefined;
    if (t) out.push(['wall, median (mm)', num(t.median)]);
  }
  if (d.yield_pct !== undefined) out.push(['yield (%)', num(d.yield_pct, 1)]);
  if (d.pour_mass_kg !== undefined) out.push(['pour mass (kg)', num(d.pour_mass_kg, 3)]);
  return out;
}
