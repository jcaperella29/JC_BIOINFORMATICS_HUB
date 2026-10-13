# JCAP hosted demo runner — implementation contract

The `index.html` frontend is ready to call a real hosted service. It does **not** execute Nextflow in the browser, and no backend is included in this package.

## Endpoints (same site origin)

- `POST /api/jobs` with JSON `{ "demo": "brca" | "kang" | "giladi" }`. Return JSON `{ "job_id": "opaque-safe-id" }` with HTTP 200/202.
- `GET /api/jobs/{job_id}` returns JSON `{ "status": "queued" | "starting" | "running" | "complete" | "failed" | "cancelled", "message": "optional detail", "bundle_url": "/api/jobs/{job_id}/bundle", "report_url": "/api/jobs/{job_id}/report" }`. The URL fields are mandatory on completed runs for bundle, optional for report. The frontend polls every five seconds.
- `GET /api/jobs/{job_id}/bundle` responds with a ZIP containing **every** pipeline output and relevant manifest/log; include `Content-Disposition: attachment`.
- `GET /api/jobs/{job_id}/report` returns the final human-readable report, preferably a browser-readable PDF or HTML.

## Deployment requirements

1. Deploy a separate job service that launches the **real canonical JCAP** Nextflow workflows against *fixed trusted* dataset presets. Never pass user-controlled CLI flags, local paths, or arbitrary commands to a shell.
2. Prepare the BRCA data and workflow preset (airway is the current bundled bulk example); package Kang and Giladi presets and ensure reproducible data paths.
3. Assign each run a unique isolated output directory; queue and cap concurrent work; control CPU, RAM, duration, storage, external API budget, and access to downloads; purge jobs after a stated retention period.
4. Verify that the full result bundle includes raw generated tables, enrichment, network and visualization files, literature evidence, JSON/Markdown/PDF report where generated, and execution logs/manifests. Don't exclude outputs silently.
5. Secure the runner (allowlist three demos, rate limit, quota, CORS/CSRF protections as appropriate, and avoid making paid API keys public). Return sanitized job messages. Use HTTPS.
6. Deploy via a same-origin `/api/*` route or reverse proxy. If the API is hosted elsewhere, update `JCAP_API_BASE` in `index.html` and configure CORS and the URL validation scheme accordingly.
7. Test the full run/failed run/download flow end to end before advertising one-click live execution.

The website repo holds the HTML and testing logs; the computational JCAP repo holds the workflows and data. Do not copy bulky test data or credentials into the public static website repo.
