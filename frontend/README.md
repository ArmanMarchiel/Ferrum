# frontend

React + TypeScript, built with Vite.

```bash
npm install
npm run dev        # http://127.0.0.1:5173, API proxied to :8000
npm run build      # typecheck, then bundle into dist/
npm run typecheck
```

The backend must be running for anything that touches the API:

```bash
cd ../backend/src && ../../.venv/bin/python -m uvicorn server.app:app --reload
```

## Layout

| path | what lives there |
|---|---|
| `src/pages/` | `Home.tsx` (the filesystem) and `Editor.tsx` (the CAD viewer) |
| `src/components/` | toolbar, sidebar, viewport overlays, the shared dialog |
| `src/hooks/` | `useScene` (three.js lifecycle), `useHistory` (undo/redo), `useJob` (build polling) |
| `src/services/api.ts` | every call to the backend, so one dead server surfaces as one readable error |
| `src/types/api.ts` | the backend's wire contract, hand-written to match `server/app.py` |
| `src/styles/styles.css` | one stylesheet for both pages |
| `public/` | static assets served as-is |

## Two things worth knowing

**The page CSS is scoped by a body class.** `styles.css` keeps each page's
rules under `.page-home` / `.page-editor` because their `button` baselines
genuinely disagree — the editor's is a full-width solid navy, the filesystem's
is a bare element styled by `.btn`. Neither can be hoisted to the root without
breaking the other. `usePageClass` puts the class on `<body>`.

**three.js lives outside React.** The scene is a mutable object graph driven at
60 fps; putting meshes or the camera into component state would re-render the
tree on every orbit frame. `useScene` owns it and hands back a ref, so the
components only ever call methods on it.
