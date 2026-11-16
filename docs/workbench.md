# Local search workbench

Run `python tools/serve_workbench.py` from the repository after installing Vortex.
Open `http://127.0.0.1:8765`; `--port` selects another port. The server needs only
the package's existing Python/NumPy dependencies and its compiled native module.
The frontend uses HTML, CSS and browser JavaScript without a bundler or CDN.

## Views

- Search playground: exact or IVF retrieval with top-k and probe controls, native
  timings, candidate counts, measured neighbor recall against exact retrieval,
  document previews, query highlighting and JSON export.
- Collection: a filterable list of indexed Markdown chunks with their source
  paths, starting lines and string-encoded 64-bit IDs.
- Benchmarks: archived 1m/10m capacity results, an interactive latency/recall
  chart, probe inspection and the retained full-list timing-gap diagnostic.
- Architecture: the repository's editable system diagram and the boundaries
  between encoders, native retrieval, storage and document lookup.

The source documents are loaded at startup. Restart the server to index changes
to the Markdown files. The workbench has no upload, editing or index-deletion API.

## What live search means here

The server chunks allowlisted repository Markdown files, skips fenced code, encodes
them with the existing 256D `FixtureEmbedder`, and constructs a native cosine
index with up to eight seeded IVF lists. Each request calls the actual C++
`Index.search`. Approximate results are compared with a fresh native exhaustive
query to calculate neighbor Recall@k; that validation call is outside the
displayed search timing. Independent float64 cosine checks cover this path in
the server tests.

The embedder hashes tokens. It is an offline integration fixture, not a
pretrained semantic encoder. The map uses a true PCA projection of those vectors;
its two-dimensional distances do not determine the full-vector ranking. There
is no text generation, external model endpoint or download in this interface.

The live collection is small and in memory. The 10m collection is not opened by
the workbench. Capacity figures come from the published archive, remain labeled
recorded, use a different dimension/metric, and retain their methodological
limits. The interrupted all-list point is shown with a warning, not hidden.

## HTTP boundary

The local server exposes:

| Route | Behavior |
| --- | --- |
| `GET /api/bootstrap` | Native engine identity, collection metadata, documentation chunks and PCA coordinates |
| `POST /api/search` | JSON query, strategy, top-k and probes; native results and exact-neighbor recall |
| `GET /api/benchmarks` | Read-only projections of the published capacity evidence |
| `GET /source/<allowlisted path>` | Plain-text source document |
| `GET /architecture.svg` | Existing system architecture diagram |

Static assets and source paths are explicitly allowlisted. Search bodies are
limited to 16 KiB and queries to 2,000 characters. The server accepts local Host
headers and same-origin browser search requests. It binds to IPv4 loopback;
there is no authentication or public hosting support. Search text is not logged.
Recent queries live in browser local storage and can be cleared in the sidebar.

## Accessibility and interaction

The layout adapts from desktop columns to a phone layout with a dismissible
navigation drawer. Forms have explicit labels, dialogs use the native focus trap,
controls show focus indicators, and reduced-motion preferences are respected.
Use Ctrl+K to focus the query, Ctrl+Enter to search, and Escape to close a preview
or navigation. Use left/right arrows on the vector map to move between points;
Enter opens their document. Chart probes also support keyboard activation.

## Verification

`python -m pytest tests/test_workbench.py -q` checks native results against an
independent float64 cosine reference, all-list agreement, measured partial-probe
recall, 64-bit ID preservation, PCA/source consistency, static delivery, request
validation, path restrictions and benchmark diagnostic preservation.

`node --check frontend/app.js` checks JavaScript syntax when Node.js is available.
The runtime interface itself does not require Node.js. Browser verification
covers real searches, control changes, previews, collection filtering, empty
states, recorded probe/size selection, keyboard interaction, result export and
desktop/phone layouts. New UI changes are separate from the frozen experiments;
their code, inputs and measurements remain unchanged.
