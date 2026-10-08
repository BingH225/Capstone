# SmartStress web app

React + Vite frontend, migrated into the Capstone monorepo. Use Node 22.

```bash
npm ci
npm run dev
npm run build
```

Run the Python API from the repository root (`smartstress-server`). Vite proxies `/api` and `/health` to `http://localhost:8000`; set `VITE_API_BASE_URL` only when another API origin is needed. Production requests use the same origin by default, and the Python server can host `apps/web/dist`.

The chat starts a valid text-only session and uses the typed session API. Dashboard/chart data and prediction/report examples remain mock fixtures. The planned reliability status/evidence/consent UI is not yet integrated; see `../../docs/DEVELOPMENT.md`.

Credentials belong on the backend. Do not add `VITE_*` LLM API keys: Vite variables are exposed to browsers.
