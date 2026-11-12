# Frontend Testing

## Overview
This document describes the automated test suite of the frontend (`frontend/`): the tools, the layout, the shared test infrastructure, every test case and the reason it exists, and the conventions for adding new tests.

The suite has three layers:

| layer | tool | runs in | scope | location |
|-------|------|---------|-------|----------|
| unit | Vitest + Testing Library | jsdom (simulated DOM in Node) | one function, hook or component in isolation | next to the source file, `*.test.ts(x)` |
| integration | Vitest + Testing Library | jsdom | the real app on its real routes: page, query cache, mappers and cards together; only `fetch` and 3Dmol are replaced | `src/test/integration/` |
| end-to-end (E2E) | Playwright | headless Chromium against a production build | user journeys in a real browser with real CSS, WebGL and 3Dmol; the API is mocked at the network layer | `e2e/` |

Coverage is enforced by Vitest: lines, statements and functions must stay at or above 80 %, branches at or above 75 %. The current suite measures about 99 % lines and 94 % branches.

## Running the tests
All commands run from `frontend/`.

| command | purpose |
|---------|---------|
| `npm test` | Vitest in watch mode (unit + integration) |
| `npm run test:coverage` | single Vitest run with coverage; fails below the thresholds. HTML report: `coverage/index.html` |
| `npm run test:e2e` | Playwright suite against a mocked API. Builds the app and serves it with `vite preview` automatically |
| `npm run test:e2e:fullstack` | the `@fullstack` smoke test against the real backend (see [Full-stack smoke test](#full-stack-smoke-test)) |
| `npx playwright test --ui` | Playwright's interactive runner, for debugging E2E tests |

First-time setup for E2E: `npx playwright install chromium`.

Generated output (`coverage/`, `playwright-report/`, `test-results/`) is git-ignored.

## Tooling

| package | role |
|---------|------|
| `vitest`, `@vitest/coverage-v8` | test runner (shares `vite.config.ts` with the app) and V8-based coverage |
| `jsdom` | DOM implementation for unit and integration tests |
| `@testing-library/react` | renders components and queries them the way a user perceives them (role, label, text) |
| `@testing-library/user-event` | realistic interactions (typing, clicking, keyboard) |
| `@testing-library/jest-dom` | DOM matchers such as `toBeInTheDocument`, `toBeDisabled`, `toHaveAttribute` |
| `@playwright/test` | E2E runner and browser automation |

Not used deliberately:
- **MSW:** the app talks to a single endpoint, so stubbing `fetch` (Vitest) and `page.route` (Playwright) is enough.
- **Snapshot tests:** they assert markup rather than behaviour and break on every styling change.

Node 22.12 or newer is required (Vitest 5 and jsdom); CI uses Node 24.

## Configuration

### Vitest (`frontend/vite.config.ts`, `test` block)
| setting | value | reason |
|---------|-------|--------|
| `environment` | `jsdom` | components need a DOM |
| `setupFiles` | `./src/test/setup.ts` | global mocks and cleanup, see [Test infrastructure](#test-infrastructure) |
| `include` | `src/**/*.test.{ts,tsx}` | keeps Playwright specs in `e2e/` out of Vitest |
| `css` | `false` | CSS is irrelevant in jsdom; skipping it speeds up runs |
| `coverage.include` | `src/**/*.{ts,tsx}` | measures all application code |
| `coverage.exclude` | `src/main.tsx`, `src/types/**`, `src/test/**`, `**/*.test.*` | entry point, type-only files and test code are not measured |
| `coverage.thresholds` | lines/statements/functions 80, branches 75 | a run below these fails, locally and in CI |

### Playwright (`frontend/playwright.config.ts`)
| setting | value | reason |
|---------|-------|--------|
| `testDir` | `e2e` | |
| `webServer.command` | `vite build && vite preview` on port 4173 | a production build is used because the dev server's first-load dependency optimisation reloads pages mid-test. `vite preview` inherits `server.proxy`, so `/api` still reaches the backend |
| `workers` | `2` | every page runs continuous GSAP and WebGL animation; with more parallel browsers they starve each other and page loads time out |
| `retries` | 1 in CI, 0 locally | |
| `trace` | `on-first-retry` | a failing CI run produces a step-by-step trace with screenshots |
| `projects` | Chromium only | |

### TypeScript
- Test files under `src/` are type-checked by `tsconfig.app.json` as part of `npm run build`.
- E2E specs are type-checked by `tsconfig.e2e.json`, which is referenced from `tsconfig.json`.
- `playwright.config.ts` is type-checked by `tsconfig.node.json`.

## Test infrastructure
Shared code lives in `frontend/src/test/`.

### `setup.ts`
Runs before every test file:
- registers the jest-dom matchers;
- replaces the `3dmol` package with `mocks/3dmol.ts` in every test (`vi.mock`), because 3Dmol needs WebGL, which jsdom does not provide;
- before each test, sets "prefers reduced motion" to on, so GSAP tweens last 0.2 s instead of about 1 s;
- after each test, unmounts rendered components, clears the TanStack Query cache (`queryClient.clear()`) and restores stubbed globals such as `fetch`.

The query cache is a module-level singleton (`src/api/queryClient.ts`). Without the clear step, a result cached in one test would be served to the next test that uses the same SMILES.

### `media.ts`
jsdom has no `window.matchMedia`. This file installs a stub that answers only the `prefers-reduced-motion: reduce` query. `setReducedMotion(true | false)` switches the answer per test.

### `mocks/3dmol.ts`
A stand-in for the 3Dmol viewer:
- `createViewer` returns an object whose methods (`addModel`, `setStyle`, `zoomTo`, `render`, `spin`, `clear`) are `vi.fn()` spies, so tests can assert calls and arguments;
- `getModel().selectedAtoms()` returns 21 fake atoms (the size of the aspirin fixture), with atom *i* at coordinates (*i*, *i* mod 3, 0);
- `modelToScreen` maps a model point (x, y) to the screen point (10x, 10y), so overlay positions are predictable;
- every created viewer is pushed to the exported `viewers` array.

### `fixtures/aspirin.ts`
A complete `PredictionResponse` for aspirin, shared by unit, integration and E2E tests:
- `ASPIRIN_MOLBLOCK`: a real RDKit 3D conformer with 21 atoms (13 heavy atoms followed by 8 hydrogens), so the real 3Dmol in E2E tests can render it;
- six predictions chosen to hit every family branch of the mapper: two CYP, one hERG, two Tox21 (NR-/SR-) and one "Other properties" (BBBP);
- probabilities chosen so that both confidence formulas give 0.88 (true at 0.876, false at 0.123);
- atom scores for every property, with `null` for the hydrogens, matching the GNN service's output;
- two similar compounds that share one source URL, to exercise source de-duplication.

To regenerate the molblock (repository root, RDKit environment):

```bash
uv run python -c "from rdkit import Chem; from rdkit.Chem import AllChem; m=Chem.AddHs(Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')); AllChem.EmbedMolecule(m, randomSeed=42); AllChem.MMFFOptimizeMolecule(m); print(Chem.MolToMolBlock(m))"
```

### `helpers.tsx`
| helper | purpose |
|--------|---------|
| `renderWithRouter(ui)` | renders `ui` as the only route of an in-memory router; needed by components that use router APIs such as `<Form>` |
| `renderApp(url)` | renders the real application (`QueryClientProvider` + the real route table from `routes.tsx` via `routes.routes`) at `url`, using an in-memory history |
| `mockFetch(body, status)` | stubs `fetch` to answer every call with `body` and `status` |
| `mockPendingFetch()` | stubs `fetch` with a request that never resolves but rejects with an `AbortError` when its signal aborts; used to observe loading states and cancellation |
| `requestBody(fetch, n)` | parsed JSON body of the n-th recorded `fetch` call |

### E2E helpers (`e2e/fixtures.ts`)
| helper | purpose |
|--------|---------|
| `mockApi(page, { status, body, delayMs })` | intercepts `**/api/predict-properties` in the browser and answers with the aspirin fixture or the given body. Returns the list of intercepted requests |
| `analyze(page, smiles)` | fills the SMILES input and presses Enter |

## Test catalogue

### Unit tests

#### `src/api/analysis.test.ts`: API client and query definition
| case | rationale |
|------|-----------|
| POSTs the SMILES as JSON with the caller's abort signal | wrong URL, method, header or body would be rejected by the backend; the signal is required for cancellation |
| returns the parsed response on 200 | happy path |
| throws the backend's message on 422 / 503 | the backend reports errors as `{ message }`, which the UI displays as is |
| falls back to the status when the error body is not JSON | covers the `.catch(() => null)` branch, e.g. a proxy's HTML error page |
| falls back to the status when the error JSON has no message | FastAPI's own validation errors use `{ detail }` |
| `analysisQuery` keys the cache by SMILES | the key is the cache address; different SMILES must not share an entry |
| `analysisQuery` passes TanStack's abort signal on to fetch | without the signal, TanStack cannot cancel the request when the results page unmounts |

#### `src/api/mappers.test.ts`: response-to-props mapping
| case | rationale |
|------|-----------|
| groups properties into families, in order of first appearance | covers every `familyOf` branch (CYP, hERG, NR-/SR-, fallback) and the ordering guarantee |
| reports confidence in the predicted class, rounded to 2 decimals | `p` for a positive prediction, `1 − p` for a negative one |
| returns no families for no predictions | empty input |
| uses the SMILES as the name and rounds the score | the API provides no compound names yet |
| keeps the summary and deduplicates sources in first-seen order | several compounds can cite the same source |
| has no sources without compounds | empty input |

#### `src/components/common/common.test.tsx` and `src/components/layout/Navbar.test.tsx`
| case | rationale |
|------|-----------|
| `Button` forwards native button props and renders its children | `type`, `className` and `onClick` must reach the DOM button |
| `Button` does not fire clicks when disabled | |
| `Metadata` adds the given class to the base class | |
| `Metadata` has only the base class without className | guards against a trailing space in `class` |
| `Section` renders a `<section>` with the given class | |
| `Navbar` shows the title and calls `onNewAnalysis` on click | |

#### `src/features/smiles-input/SmilesInput.test.tsx`
`SmilesInput` is a controlled component, so the tests render it inside a small host that keeps the value in `useState`, as `AnalysisPage` does.

| case | rationale |
|------|-----------|
| labels the input | the label's `htmlFor` must match the input's `id` for assistive technology |
| reports every keystroke through `onChange` | controlled-input contract |
| submits the trimmed SMILES from the button | surrounding whitespace is removed before analysis |
| submits on Enter | keyboard use |
| ignores a blank submit (empty and whitespace-only) | guard clause |
| disables the button when disabled | the input is disabled while results are shown |

#### `src/features/property-view/PropertyView.test.tsx`
| case | rationale |
|------|-----------|
| `PropertyItem` shows the name, the predicted class and the confidence | |
| `PropertyItem` shows False for a negative prediction | `data-value="false"` drives the styling |
| `PropertyItem` carries the `data-*` hooks the motion module counts up from | `data-animate="count"` and `data-value` are part of the selector contract in [frontend-styling.md](./frontend-styling.md#selector-contract) |
| `PropertyItem` is pressed when selected and reports its name on click | `aria-pressed` reflects the selection |
| `PropertyItem` tolerates a click without a handler | optional callback |
| `PropertyResultsContainer` renders one item per result and presses only the selected one | |
| `PropertyPredictionCard` shows an empty state without predictions | |
| `PropertyPredictionCard` starts on the first family and offers every family | |
| `PropertyPredictionCard` swaps the list when another family is chosen | |
| `PropertyPredictionCard` passes the selection through to the items | |

#### `src/features/similar-molecules`, `scientific-interpretation`, `molecule-overview`
| case | rationale |
|------|-----------|
| `SimilarMoleculesCard` renders nothing without similar molecules | the card is omitted, not shown empty |
| `SimilarMoleculesCard` lists each molecule with its score, plus the footnote | |
| `ScientificInterpretationCard` renders nothing without an interpretation | |
| `ScientificInterpretationCard` shows numbered source links that open in a new tab | links must carry `target="_blank"` and `rel="noopener noreferrer"`; without `noopener` the opened page could control the app's tab |
| `ScientificInterpretationCard` shows no links without sources | |
| `MoleculeOverviewCard` renders the formula with subscripted counts | `C9H8O4` renders 9, 8 and 4 in `<sub>` |
| `MoleculeOverviewCard` lists the descriptors | molar mass to two decimals with unit, atom and bond counts, SMILES |
| `MoleculeOverviewCard` renders the viewer it is given and the 4-element legend | |

#### `src/features/molecule-overview/MoleculeViewer.test.tsx`: 3D viewer with mocked 3Dmol
| case | rationale |
|------|-----------|
| loads the molblock and renders it once | `addModel(molblock, "sdf")`, `setStyle`, `zoomTo` and `render` are each called once |
| spins by default and the toggle pauses and resumes it | toggle state, `aria-pressed`, accessible name, `spin("y")` and `spin(false)` |
| starts paused when the user prefers reduced motion | accessibility |
| shows no attribution without scores | default accessible name, no attribution bar, empty overlay |
| marks every scored atom and labels only the four strongest | one ring per non-null score (13), labels only for the top four by magnitude, label format such as `+1.2e-1`, positions taken from `modelToScreen` |
| handles all-zero scores without NaN | the normalisation divides by the largest magnitude; the `\|\| 1` fallback must prevent `NaN` |
| clears the marks when the scores go away | effect cleanup |
| tears the viewer down on unmount | `spin(false)`, `clear()`, canvas removed, animation frame cancelled; prevents leaks and stacked canvases |
| builds a new viewer for a new molblock | the setup effect depends on `molblock` |

#### `src/styles/molecule3d.test.ts`: decorative molecule renderer
The first three cases run once for each of the eight molecules in `MOLECULES`.

| case | rationale |
|------|-----------|
| draws one sphere per atom and one bond group per bond, double bonds as two lines | geometry and DOM structure; compared as sorted counts because the scene is depth-sorted |
| projects every atom inside the viewBox | checks centring and normalisation (\|x\|, \|y\| < 1) |
| shows the attribution highlight when not animated | static mode shows the first highlight group at full strength |
| gives each mount its own gradient ids | mounts must not overwrite each other's gradients |
| removes its svg on dispose | |
| rotates on each ticker frame while animated and stops after dispose | the GSAP ticker drives rotation; after `dispose()` the ticker callback must be gone |

#### `src/styles/loader.test.ts`: analysis loader
| case | rationale |
|------|-----------|
| draws every heavy atom and leaves hydrogens out | the graph is a heavy-atom sketch; the 4 oxygens of aspirin carry labels |
| keeps every atom inside the viewBox | layout and scaling for all three sketchable molecules |
| at each phase's time shows that phase: its bar and caption together, nothing else | runs at a point inside each of the 4 phases; the bar and caption must always agree |
| keeps bar and caption in step on the second loop too | the timeline repeats; sync must survive the restart |
| holds the attribution frame without motion | reduced motion: paused timeline, phase 3 caption, halos drawn |
| shows the SMILES and elapsed time, and flags a slow request | `performance.now` is mocked; after 12 s the host gets `data-slow` |
| cleans up by itself once its host leaves the page | React removes the host without notice; the next ticker frame removes the generated DOM and the timeline |
| leaves the host's own content alone | the React-owned status `<p>` stays the only accessible child |

#### `src/styles/motion.test.tsx`: `useAnalysisMotion`
The hook runs against a DOM built from the selectors it targets (see the selector contract in [frontend-styling.md](./frontend-styling.md#selector-contract)). The first six cases run with reduced motion on and off.

| case | rationale |
|------|-----------|
| mounts the background molecules into known slots and disposes them on unmount | unknown `data-molecule` values are skipped |
| `exitLanding` resolves once the landing copy has left | navigation waits on this promise |
| `exitLanding` resolves at once when there is no landing copy | a missing target must not block navigation |
| `exitResults` resolves once the results have left | |
| mounts the analysis loader whether it is already there or appears later, and removes it on unmount | both mount paths (initial query and `MutationObserver`) and the dispose on unmount |
| `exitResults` resolves at once when there are no results | |
| counts numbers up from zero when a list mounts later | the `MutationObserver` path for lists that appear after data loads |
| re-staggers a card's list when its family select changes | the delegated `change` listener |
| ignores change events from other elements | |

#### `src/App.test.tsx`
| case | rationale |
|------|-----------|
| mounts the router on the landing page | smoke test of the provider and router setup |

### Integration tests

#### `src/test/integration/analysis-flow.test.tsx`
| case | rationale |
|------|-----------|
| starts on the landing page without results | |
| submits a SMILES, shows the loading state, then every card | URL becomes `/CCO`; exactly one request with the trimmed SMILES; analysis loader with `aria-busy` (checked on `.closest(".analysis-loader")`, because the text sits in its status `<p>`); landing hero removed; analyze button disabled |
| renders the mapped response | API → mapper → card wiring: molar mass, default family, confidence, similar molecules, summary, de-duplicated sources (3 → 2), viewer |
| decodes a deep link before sending it | `/CC(%3DO)O` sends `CC(=O)O` |
| toggles a property's atom attribution, and keeps it across families | the selection lives in `AnalysisResults` and is shared by the property card and the viewer |
| New Analysis returns to an empty landing page | input cleared, hero shown, button enabled |
| analyzing the same molecule again is served from the cache | the TanStack Query cache (`staleTime: Infinity`) prevents a second request |
| New Analysis while loading aborts the pending request | leaving the results page unmounts the last query observer, and TanStack aborts the request |
| New Analysis does nothing on the landing page | |
| leaves out the similar-molecules and interpretation cards when the response has none | |

#### `src/test/integration/analysis-errors.test.tsx`
| case | rationale |
|------|-----------|
| shows the 422 / 503 / 500 error in an alert | the message appears in `role="alert"`, no result cards render, and only one request is sent (`retry: false`) |
| shows a network failure in an alert | a rejected `fetch` surfaces its message |

### End-to-end tests (`e2e/`)

| spec | case | rationale |
|------|------|-----------|
| `landing.spec.ts` | shows the hero, the SMILES input and the analyze button | |
| | a blank submit (empty and whitespace-only) stays on the landing page | URL unchanged, no request sent |
| `analysis.spec.ts` | analyzes a molecule end to end | URL is percent-encoded; request body holds the trimmed SMILES; loading state, then formula, property family, similar molecules, summary; 3Dmol renders a real WebGL `<canvas>` |
| | selecting a property shows its atom attribution, selecting it again hides it | `aria-pressed`, attribution bar, accessible name of the canvas, 13 overlay rings |
| | the family select swaps the property rows | |
| | the rotation toggle pauses and resumes | |
| | source links open in a new tab | `target` and `rel` attributes |
| | works with reduced motion, and the model starts paused | `page.emulateMedia({ reducedMotion: "reduce" })` |
| `errors.spec.ts` | a 422 / 503 / 500 response shows its message | |
| | a network failure shows an alert | `route.abort()` |
| `navigation.spec.ts` | a shared link or refresh loads that molecule's results | direct URL and reload both work; the in-memory cache is lost on reload, so two requests are sent |
| | back returns to the landing page and forward to the cached results | browser history works; forward is served from the cache (one request in total) |
| | New Analysis returns to an empty landing page | |
| `fullstack.spec.ts` | aspirin gets real predictions (`@fullstack`) | see below |

Some E2E cases repeat unit-level checks on purpose: unit tests run in jsdom with a mocked 3Dmol, while E2E proves the same behaviour with real CSS, real WebGL and a real browser.

#### Full-stack smoke test
`fullstack.spec.ts` is tagged `@fullstack`. It is excluded from `npm run test:e2e` and from CI. It requires:
- the database (`docker compose up -d db`),
- the GNN service,
- the backend app on port 8001 (`vite preview` proxies `/api` there).

It analyzes aspirin, waits up to 60 seconds for the results (90-second test timeout), and checks only the shape of the result (cards present, at least one property and one similar molecule, a rendered canvas, no alert), not the values, because model outputs change between versions.

## Continuous integration
`.github/workflows/ci.yml`:

| job | steps |
|-----|-------|
| `frontend-ci` | `npm ci` → `npm run lint` → `npm run test:coverage` (fails below the thresholds) → `npm run build` (also type-checks all test code) |
| `frontend-e2e` | `npm ci` → `npx playwright install --with-deps chromium` → `npm run test:e2e`; on failure, `playwright-report/` is uploaded as an artifact (kept 7 days) |

Both jobs run on Node 24 with read-only token permissions.

## Writing tests

### Conventions
- **File placement:** unit tests sit next to the file they test (`Foo.tsx` → `Foo.test.tsx`). Cross-module flows go to `src/test/integration/`; browser journeys go to `e2e/`.
- **Query by role, label or text** (`getByRole("button", { name: "Analyze Molecule" })`), not by class or test id. This keeps tests independent of styling, and a missing accessible name fails the test, which also catches accessibility regressions. Class selectors are acceptable only where the class itself is the contract (the motion selectors, `.molecule-overview__overlay`).
- **Interact through `userEvent`**, not `fireEvent`, so focus, keyboard and pointer events happen as in a browser.
- **Use `findBy…` / `waitFor` for anything asynchronous.** This includes data loading, lazy components and elements that fade in from `visibility: hidden` (GSAP `autoAlpha`), which Testing Library treats as inaccessible until visible.
- **Never call the real network.** Stub `fetch` with the helpers in Vitest, or use `mockApi` / `page.route` in Playwright.
- **Reuse `fixtures/aspirin.ts`** instead of inventing new response shapes, and extend it when a new field is added to `PredictionResponse`.
- **Known defects** may be documented with `it.fails(...)` and a comment describing the defect. When the defect is fixed, the test starts failing; remove `.fails` at that point.

### Pitfalls
| symptom | cause and fix |
|---------|---------------|
| `getByRole` cannot find a visible heading | the element is still faded out by GSAP; use `findByRole` |
| a test sees results from another test | the query cache was not cleared; `setup.ts` handles this for tests using the shared `queryClient`, so keep using it |
| a router assertion passes but a side effect has not happened yet | `router.state` updates as soon as `navigate()` runs, before React re-renders and unmounts; wrap the follow-up assertion in `waitFor` |
| E2E `page.goto` times out | too many parallel browsers; keep `workers` at 2 |
| a new component crashes in jsdom with `matchMedia is not a function` | the test file bypassed `setup.ts`; tests must live under `src/` and match `*.test.ts(x)` |
| a GSAP-driven assertion is flaky | assert the end state with `waitFor`, not intermediate values. Exiting a view immediately after mounting races its entrance tween |
