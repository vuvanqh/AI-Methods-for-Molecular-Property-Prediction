# Frontend Styling and Motion

## Overview
This document describes the styling layer and the GSAP motion of the frontend, as of the `feat/frontend` branch. The visual target is the reference GUI in `diagrams/gui.png`: a dark, research-style interface.

The division of work is strict:

- **React components** (markup, state, routing, data flow) are written by hand.
- **The styling layer** (CSS, design tokens, fonts, GSAP motion) hooks into components only through `className`, `id` and `data-*` attributes, plus one `import "./X.css"` line per component.

As a result, no component contains animation code. Motion is driven from one hook (`useAnalysisMotion`), which finds elements by the selectors listed in [Selector contract](#selector-contract). Renaming a class in a component can silently disable a style or an animation, so check this document before renaming.

### File map

| file | role |
|------|------|
| `frontend/src/index.css` | fonts, design tokens, base styles, shared utility classes |
| `frontend/src/styles/motion.ts` | all page motion: the `useAnalysisMotion` hook |
| `frontend/src/styles/molecule3d.ts` | decorative SVG 3D molecule renderer for the landing page |
| `frontend/src/styles/loader.ts` | the analysis loader: a GNN pipeline sketch shown while the request is pending |
| `frontend/src/pages/AnalysisPage.css` | page shell: landing and results layout, molecule field, results heading, card grid |
| `frontend/src/components/layout/Navbar.css` | sticky top bar |
| `frontend/src/features/smiles-input/SmilesInput.css` | SMILES input bar |
| `frontend/src/features/molecule-overview/MoleculeOverview.css` | 3D/structural overview card (`MoleculeOverviewCard.tsx`, `MoleculeViewer.tsx`), including its loading and error placeholder |
| `frontend/src/features/property-view/PropertyView.css` | property prediction card |
| `frontend/src/features/similar-molecules/SimilarMolecules.css` | similar molecules card |
| `frontend/src/features/scientific-interpretation/ScientificInterpretation.css` | scientific interpretation card |

## Dependencies

| package | version | purpose |
|---------|---------|---------|
| `gsap` | ^3.15.0 | animation engine, including the `Flip` plugin |
| `@gsap/react` | ^2.1.2 | `useGSAP` hook: scoped animations with automatic cleanup on unmount |
| `@fontsource-variable/plus-jakarta-sans` | ^5.3.0 | UI typeface (variable weight), self-hosted |
| `@fontsource/jetbrains-mono` | ^5.3.0 | monospace for SMILES, scores and confidences (weights 400 and 500), self-hosted |

The fonts are imported at the top of `index.css`, so `main.tsx` does not need to change and no CDN is used. `index.html` sets the page title to "Molecular Property Analysis".

## Design tokens
All tokens are CSS custom properties on `:root` in `index.css`. The theme is dark-only (`color-scheme: dark`). Use tokens rather than raw values in new CSS.

### Colour

| token | value | use |
|-------|-------|-----|
| `--bg` | `#0b1220` | page background |
| `--bg-deep` | `#080e1a` | inset surfaces (input field, 3D viewer) |
| `--surface` | `#0f1729` | card body |
| `--surface-2` | `#131d31` | card top (cards use a `--surface-2` → `--surface` gradient) |
| `--surface-3` | `#18233a` | raised elements |
| `--border` | `#1e2a40` | hairlines, card borders |
| `--border-strong` | `#2a3852` | input, button and chip borders |
| `--text` | `#e6eaf2` | primary text |
| `--text-muted` | `#8a96ab` | secondary text, labels |
| `--text-subtle` | `#5d6a82` | placeholders, empty states |
| `--accent` | `#b8a6f4` | primary button, focus ring (lavender) |
| `--accent-hover` | `#c7b8f8` | primary button hover |
| `--accent-ink` | `#1a1433` | text on `--accent` |
| `--accent-2` | `#8fa8ff` | end of the headline gradient (periwinkle) |
| `--accent-soft` | `rgb(184 166 244 / 0.12)` | input focus halo |
| `--link` | `#7ea6d8` | links, sublabels, family selector |
| `--warm` | `#e7b36a` | positive predictions, status text, the SMILES label icon |
| `--warm-soft` | `rgb(231 179 106 / 0.12)` | warm tint |
| `--teal` / `--teal-soft` | `#5fd4c4` / 8 % tint | "Browser-rendered 3D" chip |
| `--yes` | `var(--warm)` | prediction = true |
| `--no` | `#6b7891` | prediction = false |
| `--atom-c` / `--atom-o` / `--atom-n` / `--atom-h` | `#9aa3b2` / `#e5484d` / `#5b7cfa` / `#f1f3f7` | atom colours; shared by the 3D molecules and the overview legend |

### Shape, spacing, type, motion

| token | value |
|-------|-------|
| `--radius` / `--radius-sm` / `--radius-xs` | 12px / 8px / 6px |
| `--hairline` | `1px solid var(--border)` |
| `--inset-hl` | 1px top inner highlight used on cards |
| `--shadow-card` | soft drop shadow used on cards |
| `--s-1` … `--s-16` | spacing scale: 4, 8, 12, 16, 20, 24, 32, 40, 48, 64px (`--s-1`, `-2`, `-3`, `-4`, `-5`, `-6`, `-8`, `-10`, `-12`, `-16`) |
| `--sans` | Plus Jakarta Sans Variable, with system fallbacks |
| `--mono` | JetBrains Mono, with system fallbacks |
| `--ease-out` | `cubic-bezier(0.22, 1, 0.36, 1)`, used for CSS hover and focus transitions |
| `--t-fast` | 180ms, the duration of all CSS hover and focus transitions |

The base font is 15px/1.55 Plus Jakarta Sans.

## Base styles
`index.css` also sets the following:

- `box-sizing: border-box` everywhere.
- A thin dark scrollbar.
- The body background: `--bg` with a faint blue radial glow at the top.
- `#root` as a full-height flex column. The page layout depends on this (see [Layout states](#layout-states)).
- Default margins removed from headings, paragraphs and lists.
- `h1`: clamp(28–36px), weight 650. `h2`: 15px, weight 650.
- Buttons and inputs inherit the font.
- A lavender `:focus-visible` ring and a lavender `::selection` colour.

## Utility classes
These are shared classes from `index.css` that any component can use.

| class | effect |
|-------|--------|
| `.container` | centred column, max 1096px with 24px gutters (16px under 600px) |
| `.card` | surface gradient, hairline border, 12px radius, inset highlight and shadow. **Motion also uses this class**: every `.card` inside `.analysis-results` is animated in. |
| `.card__header` / `.card__footer` | standard card padding |
| `.eyebrow` | 10.5px uppercase label with wide tracking |
| `.metadata` | 12px muted text |
| `.gradient-text` | lavender → periwinkle gradient clipped to the text |
| `.mono` | monospace font |
| `.chip`, `.chip--teal` | 32px pill label; the teal variant is the "Browser-rendered 3D" style |
| `.btn`, `.btn--primary`, `.btn--ghost` | buttons: lavender filled and outline variants, with hover, active (1px press) and disabled states |
| `.table-head` | column header row with a bottom hairline |
| `.row` | hairline-divided list row |
| `.dot`, `.dot--yes`, `.dot--no` | 6px status dot; the yes variant glows |
| `.empty-state` | centred muted text for empty lists |

## Layout states
The analysis page has two visual states. CSS tells them apart by which elements are present, using `:has()`, so no layout class has to be toggled from React.

| state | detected by | layout |
|-------|-------------|--------|
| Landing | `.landing-hero` exists | copy (720px) and SMILES bar (760px) centred vertically; 3D molecules float in the side gutters |
| Results | `.analysis-results` exists | heading, then cards, then the SMILES bar at the bottom (1048px wide) |
| Neither (no landing copy yet) | `#root:not(:has(.landing-hero))` | the SMILES bar is centred vertically on its own |

Breakpoints:

| width | change |
|-------|--------|
| ≤ 1280px / ≤ 1100px / ≤ 960px | fewer decorative molecules, then none (see [Slot placement](#slot-placement)) |
| ≤ 880px | the two-column card row (`.analysis-page-tier2`) and the overview card body stack into one column |
| ≤ 600px | page widths become `100% − 32px`; the SMILES bar stacks (input above button); the navbar shrinks to 60px |

## Component stylesheets

### Navbar (`Navbar.css`)
- `.navbar`:
  - Sticky, 70px tall, with a translucent background and a 14px backdrop blur.
  - `padding-inline: max(24px, (100% − 1048px) / 2)` lines the title up with the page content without needing a wrapper element.
- `.navbar-title`: 17px, bold.
- `.navbar-actions`: flex row for buttons.
- `.navbar-button`: combined with `.btn .btn--ghost`.

### SMILES input (`SmilesInput.css`)
- `.smiles-input-container` (combined with `.card`): the bar's padding.
- `.smiles-input-label`: a label row. The file-code icon is drawn in CSS (`::before` with a mask), coloured `--warm`.
- `#smiles-input`:
  - 48px tall, monospace, with a deep inset background.
  - Lavender border and halo on focus.
  - Muted text when disabled.
- `.smiles-input-button` (combined with `.btn .btn--primary`): at least 172px wide. An arrow is added with `::after` and slides 3px on hover.

### Molecule overview (`MoleculeOverview.css`)
Used by `MoleculeOverviewCard` and `MoleculeViewer`. The header comment of the file contains the full markup (the class contract). In summary:

- `__header`: name and subtitle on the left; formula chip, divider and "Browser-rendered 3D" chip on the right.
- `__body`: a 1.45fr / 1fr grid.
  - **Viewer:** `__viewer` is a dark inset panel with a dot grid. Inside it:
    - `__canvas` is an absolutely positioned slot for the 3Dmol.js canvas.
    - `__viewer-caption` sits top-left; `__toolbar` and `__tool` (30px square buttons) sit top-right.
    - `__credit` sits bottom-right.
  - **Panel:** `__panel` holds:
    - an `.eyebrow`;
    - `__title`, which contains a `--warm` `__status` line;
    - `__description`;
    - `__stats`, a 2×2 `<dl>` with hairline dividers, styled with `__stat-label` and `__stat-value`;
    - `__legend`, whose `li[data-atom="C|O|N|H"]` items get a coloured dot from the atom tokens.
- `.molecule-overview--placeholder`: the 340px dot-grid panel. While the analysis loads it also carries `.analysis-loader` (see [Analysis loader](#analysis-loader)); when the analysis fails it shows the API's message (`role="alert"`).
- The loaded card fades and rises in (0.6s CSS animation), so it doesn't pop in over the placeholder.
- The 3Dmol viewer reads its atom colours from the `--atom-*` tokens, so the model and the legend always match.

### Property prediction (`PropertyView.css`)
- `.property-prediction-header`: the title, with the family selector stacked beneath it.
- `.property-family-select`:
  - The native `<select>` restyled as the reference's inline "Method ⌄" control: blue text, a chevron SVG background, and a faint tint on hover.
  - Changing it also re-runs the row animation (see [Family selector change](#family-selector-change)).
- `.property-button`: a full-width row laid out as a grid `name | value | confidence`, with a hairline and hover tint. On mobile the confidence wraps to its own line.
- `.property-value`:
  - The dot comes from `::before`; the colour comes from `data-value`.
  - `data-value="true"` gives warm text and a glowing dot. Anything else gives a grey dot.
- `.property-confidence`: 11px monospace, right-aligned, truncated with an ellipsis.

### Similar molecules (`SimilarMolecules.css`)
- `.similar-molecules-card`: a flex column, so the footnote sits at the bottom when the card is stretched to the height of its neighbour.
- Table-head labels are in the link colour.
- `.similar-molecule-item`: a row with hover tint.
- `.similar-molecule-name`: 13.5px semibold, truncated. An optional `<small>` inside it renders as a monospace blue subline (for example the SMILES).
- `.similar-molecule-score`: 19px monospace with tabular figures, so the digits don't shift while counting up.
- `.similar-molecules-footnote`: small blue text pinned to the bottom.

### Scientific interpretation (`ScientificInterpretation.css`)
- `.scientific-interpretation-header`: inset hairline below the title.
- `.explanation`:
  - 14px, line-height 1.75, spans the full card width.
  - Paragraphs use `white-space: pre-line`, so line breaks in the model output are kept.
- `.sources`: a wrapping row of citation chips. It is hidden when empty.
- `.source a`: a monospace chip; `[` and `]` are drawn with CSS.

### Page shell (`AnalysisPage.css`)
- **Landing:** `.landing-hero` is the centred column (max 720px) for the landing copy:
  - `.eyebrow`;
  - an `h1` of clamp(34–52px), which can contain a `.gradient-text` span;
  - a subtitle paragraph.
  - A faint lavender radial glow sits behind it.
- **Molecule field:**
  - `.molecule-field`: a fixed, non-interactive layer behind the content.
  - `.molecule-field__item`: the ten slots described in [Decorative 3D molecules](#decorative-3d-molecules).
- **Results:**
  - `.analysis-results`: a 1048px column with 20px gaps.
  - `.analysis-hero`: the results heading (eyebrow, h2, `.analysis-hero-subtitle`).
  - `.analysis-page-tier2`: the two-column card row (properties, similar molecules), with equal-height cards.
  - `.analysis-page-tier3`: the full-width row holding the scientific interpretation card.

## Motion (GSAP)

### Principles
- **GSAP vs. CSS:** GSAP is used only for entrances, exits and state changes. Hover and focus effects are plain CSS transitions (`--t-fast`, `--ease-out`).
- **Only cheap properties are animated:** `transform` and `opacity`, plus `filter: blur` on the few text lines. Layout properties are never animated.
- **`autoAlpha` instead of `opacity`:** this also sets `visibility: hidden` at 0, so hidden elements can't be clicked or focused mid-animation.
- **`clearProps`** removes the inline transforms and filters once an entrance finishes, so the element goes back to its CSS state.
- **No React re-renders:** all motion runs outside React. The count-up tweens the existing DOM text node instead of setting state.

### Wiring
The hook is called once, in `AnalysisPage`, which owns the routing. `analysisStarted` is derived from the URL (`useMatch("/:smiles")`), and every navigation goes through a handler that plays the exit first:

```tsx
const analysisStarted = useMatch("/:smiles") !== null;
const { exitLanding, exitResults } = useAnalysisMotion({ analysisStarted });

const analyze = async (smiles: string) => {      // SmilesInput → onAnalyze
    await exitLanding();                          // landing copy blurs out, bar position recorded
    navigate(`/${encodeURIComponent(smiles)}`);   // results route mounts → results entrance plays
};

const newAnalysis = async () => {                 // Navbar → onNewAnalysis
    if (!analysisStarted) return;
    await exitResults();                          // cards leave, bar position recorded
    setInputSmiles("");
    navigate("/");                                // landing mounts → landing entrance plays
};
```

`SmilesInput` and `Navbar` only report intent through callbacks (`onAnalyze`, `onNewAnalysis`). They don't navigate themselves, so the page can put the exit animation in front of every route change.

| option | type | notes |
|--------|------|-------|
| `analysisStarted` | `boolean` | Picks which entrance plays. It must become `true` in the same render that mounts `.analysis-results`, and `false` in the same render that mounts `.landing-hero`. Deriving it from the route guarantees that, including on refresh and back/forward. |
| `scope` | `RefObject<HTMLElement \| null>` (optional) | Limits selector lookups to this element. Defaults to `document`. The navbar is outside most page wrappers, so it only animates if it is inside the scope. |

| return | type | behaviour |
|--------|------|-----------|
| `exitLanding()` | `() => Promise<void>` | Plays the landing exit, records the SMILES bar position for the Flip glide, then resolves. |
| `exitResults()` | `() => Promise<void>` | Plays the results exit, records the SMILES bar position, then resolves. |

Always `await` the exit before navigating. Navigating first unmounts the elements before they can animate out. The browser's back and forward buttons skip the exit, because the route changes before any code runs, but the next view's entrance still plays.

### Selector contract
These are the class names and attributes that motion looks for. Removing or renaming any of them in a component disables the corresponding animation, without any error.

| selector | animation |
|----------|-----------|
| `.navbar` | fades down on first load |
| `.landing-hero > *` | each direct child is one line of landing copy: rises in, blurs out |
| `.molecule-field__item[data-molecule]` | slot that receives a 3D molecule. Valid values: any key of `MOLECULES` in `molecule3d.ts` (`aspirin`, `caffeine`, `benzene`, `ethanol`, `salicylic`, `paracetamol`, `pyridine`, `glycine`) |
| `.smiles-input-container` | glides between its landing and results positions (Flip) |
| `.analysis-hero > *` | each direct child is one line of the results heading |
| `.analysis-results .card` | result cards, staggered in order of appearance in the DOM |
| `[data-animate="stagger"]` | a list whose **children** stagger in, including lists that mount later (a `MutationObserver` reveals them when the analysis response arrives) |
| `[data-animate="count"][data-value]` | a number that counts up from 0 to `data-value` (must be inside a `stagger` list) |
| `.property-family-select` | a `change` event on it re-staggers the rows of its card |
| `[data-animate="loader"]` | receives the [analysis loader](#analysis-loader), whether it is on the page when the hook mounts (a direct link to `/:smiles`) or appears later (a `MutationObserver`). Mounted with reduced motion too, as a still frame. Optional `data-smiles` and `data-molecule` |

Where these are currently set in the components:

| component | attributes added |
|-----------|------------------|
| `PropertyResultsContainer` | `<ul className="property-list" data-animate="stagger">` |
| `PropertyItem` | `data-animate="count" data-value={property.confidence}` on `.property-confidence` |
| `SimilarMoleculesCard` | `<ul className="similar-molecules-list" data-animate="stagger">` |
| `SimilarMoleculeItem` | `data-animate="count" data-value={score}` on `.similar-molecule-score` |
| `AnalysisResultsPage` | `data-animate="loader" data-smiles={smiles}` on the loading `div`, which holds the `.analysis-loader__status` line |
| `AnalysisPage` | the `.molecule-field` block with ten `.molecule-field__item` slots, two of them with the `--backdrop` modifier (always rendered, on landing and results) |

### Timelines
All times are in seconds from the start of the step. Unless noted otherwise, the eases are `power3.out` for entrances and `power2.in` for exits.

#### 1. First load (once per page load, on any view)

| start | target | from → to | duration | stagger |
|-------|--------|-----------|----------|---------|
| 0 | `.navbar` | y −8, hidden → in place, visible | 0.6 | – |
| 0.1 | molecule slots | scale 0.9, random y ±14, hidden → scale 1, visible (`power2.out`) | 1.4 | 0.12 |

The molecules are mounted here, once, and stay mounted (rotating and floating) on both the landing and the results view.

#### 1b. Landing entrance
Runs on first load of `/`, and when `analysisStarted` returns to `false`.

| start | target | from → to | duration | stagger |
|-------|--------|-----------|----------|---------|
| 0.25 | landing copy lines | y +18, blur 6px, hidden → sharp, visible | 0.9 | 0.09 |
| 0.6 | SMILES bar, first load | y +16, hidden → visible | 0.8 | – |
| 0 | SMILES bar, coming back from results | Flip glide from the bottom position to the centre (`power3.inOut`) | 0.7 | – |

#### 2. Ambient (always)
These run for as long as the page is open:

- **Rotation and attribution highlight:** each slot gets a rotating 3D molecule with a cycling attribution highlight (see [Decorative 3D molecules](#decorative-3d-molecules)).
- **Float:** each slot also gets an endless float.
  - Movement: y ±14–22px, x ±6px.
  - Duration: 5–9 s per direction, with `sine.inOut` and `yoyo`.
  - Start delay: 1.5–3 s.
  - All values are random per slot, so the molecules never move in sync.

#### 3. Landing exit (`exitLanding()`)

| start | target | to | duration | stagger |
|-------|--------|----|----------|---------|
| 0 | landing copy lines | y −12, blur 4px, hidden | 0.45 | 0.05 |

The molecules are not touched; they keep turning in the background. When the tween completes, the hook records the SMILES bar position with `Flip.getState` and resolves the promise. Total: 0.45 s plus 0.05 s per additional line.

#### 4. Results entrance
Runs when `analysisStarted` becomes `true`.

| start | target | from → to | duration | stagger |
|-------|--------|-----------|----------|---------|
| 0 | SMILES bar | Flip glide from the recorded centre position to the bottom (`power3.inOut`) | 0.7 | – |
| 0.05 | `.analysis-hero` lines | y +18, blur 6px, hidden → in place | 0.9 | 0.09 |
| 0.15 | `.analysis-results .card` present now (the analysis loader) | y +24, hidden → in place | 0.8 | 0.08 |

When the response arrives, the real cards replace the placeholder:

| when | target | from → to | duration | stagger |
|------|--------|-----------|----------|---------|
| on mount | cards (CSS `analysis-card-in`) | y +12, transparent → in place; delay 0 / 0.08 / 0.16 s per row | 0.6 | by row |
| on mount + 0.2 | children of each `stagger` list (`MutationObserver`) | y +8, hidden → in place (`power2.out`) | 0.5 | 0.04 |
| list start + 0.1 + 0.04 × i | i-th `count` element of that list | text 0 → `data-value` (`power2.out`) | 1.1 | – |

#### 5. Results exit (`exitResults()`)
- **Targets:** the heading lines and all cards.
- **Motion:** y +16, then hidden, over 0.4 s.
- **Stagger:** 0.05 s, starting from the last element, so the bottom card leaves first.
- **Afterwards:** the bar position is recorded and the promise resolves.

#### Family selector change
When `.property-family-select` changes, React re-renders the rows first. On the next animation frame, every `stagger` list inside the same `.card` replays its row stagger and count-up. Reduced-motion users get no animation here.

### SMILES bar glide (Flip)
The same SMILES bar sits in the centre on the landing page and at the bottom on the results page. To avoid a jump between the two layouts, the exits call `Flip.getState(bar)` just before React swaps the layout. The next entrance then calls `Flip.from(state, { scale: false })`, which animates the bar from its old position to its new one with a transform.

`scale: false` makes Flip animate the bar's real width rather than a scaled image, because the bar is a different width in the two layouts (760px vs. 1048px). The recorded state is used once and then cleared.

### Count-up
`countUp` looks inside a `[data-animate="count"]` element for a text node that is purely numeric. It tweens that node from 0 to `data-value`, keeping as many decimals as `data-value` has, and puts the original text back when it finishes. Because it edits the node React created rather than replacing it, React's DOM stays consistent.

The number therefore has to be **its own JSX expression**, separate from any surrounding text:

```tsx
<span data-animate="count" data-value={x}>Confidence: {x}</span>   // works: "0.91" is its own text node
<span data-animate="count" data-value={x}>{`Confidence: ${x}`}</span> // does nothing: one mixed text node
```

### Lifecycle and cleanup
- **Two contexts:**
  - one `useGSAP` without dependencies mounts the molecules, the navbar fade and the floats once;
  - one with `dependencies: [analysisStarted]` plays the view entrances. With dependencies, `useGSAP` keeps its context for the life of the component and reverts it (killing all tweens) only on unmount.
- **Molecules are disposed on unmount only,** by the first context's cleanup. Their ticker callbacks are not part of a GSAP context, so they are removed manually: `dispose()` removes the ticker callback, kills the highlight timelines and removes the `<svg>`.
- **The exit functions are wrapped in `contextSafe`,** so tweens created on click belong to the context and are cleaned up with it.
- **React StrictMode:** the double mount in development is safe. The first run is reverted and disposed, then the second run mounts fresh molecules.
- **Listeners:** a delegated `change` listener on `document` (family selector) and a `MutationObserver` on `document.body` (lists and analysis loaders that mount later). Both are removed on unmount, and every loader the hook mounted is disposed with them. The observer skips non-HTML nodes, so the molecules' constant SVG reordering costs almost nothing.
- **Empty selections are skipped** (for example, no cards on the landing page), so GSAP never warns about missing targets.

### Reduced motion
When the OS has "reduce motion" enabled (`prefers-reduced-motion: reduce`):

- every entrance and exit becomes a 0.2 s opacity fade, with no movement, blur, Flip glide or count-up;
- the molecules are drawn once as a static 3D pose, with the first atom group highlighted, and do not rotate or float;
- the analysis loader holds its attribution frame (see [Analysis loader](#analysis-loader)); its clock still counts;
- `exitLanding` and `exitResults` still resolve, after 0.2 s;
- in CSS, `index.css` reduces all transitions and keyframe animations to 0.01 ms.

## Analysis loader
`styles/loader.ts` draws the loading state of the results page: a 2D sketch of what the GNN does, played left to right while `POST /api/predict-properties` is pending. A molecular graph draws in, message-passing pulses run along its bonds, Integrated Gradients halos mark the atoms, and the atoms pool into a molecule vector that feeds the property heads. Like the decorative molecules, it has no framework dependency and is hidden from screen readers. The only accessible content is the status line React renders.

### Markup
The page renders only the host element and its status line:

```tsx
{isPending ? <div className="molecule-overview--placeholder analysis-loader card" aria-busy="true" data-animate="loader" data-smiles={smiles}>
        <p className="analysis-loader__status" role="status">Loading molecular structure…</p>
    </div>
```

| attribute | role |
|-----------|------|
| `data-animate="loader"` | `useAnalysisMotion` mounts the loader into this element (see [Selector contract](#selector-contract)) |
| `data-smiles` | shown in the clock; optional |
| `data-molecule` | optional key of `MOLECULES`; without it, one of `aspirin`, `paracetamol` or `salicylic` is picked at random (the other molecules overlap when drawn flat) |
| `molecule-overview--placeholder` | the dot-grid panel background, and it keeps the CSS card entrance off the loader (GSAP animates it in) |

`mountLoader(host, { animate })` appends three `aria-hidden` children next to the status line and returns `{ dispose, timeline }`:

```
div.analysis-loader (React)
├─ p.analysis-loader__status            (React) "Loading molecular structure…"
├─ div.analysis-loader__stage
│  ├─ svg.analysis-loader__graph        viewBox 0 0 400 290: halos, atom-state glows, bonds, pulses, atoms
│  ├─ span.analysis-loader__flow        dashed arrow
│  ├─ div.analysis-loader__readout      "Molecule vector": 8 × 4 cells
│  ├─ span.analysis-loader__flow
│  └─ div.analysis-loader__readout      "Property heads": CYP1A2, CYP2C9, hERG, NR-AR, SR-ARE, BBBP
├─ div.analysis-loader__phase           4 bars + 4 stacked captions
└─ span.analysis-loader__clock          "<SMILES> · 4.2 s"
```

The graph uses the heavy atoms of the chosen `MOLECULES` entry (hydrogens are dropped). Molecules taller than wide are turned 90°, and the drawing is scaled to fit the viewBox (at most 56 units per Å). Each atom's halo size comes from the molecule's highlight `groups`: the first group scores 1, the second 0.65, later ones 0.4, and atoms in no group 0.15. These scores are illustrative, not model output.

### Timeline
One GSAP timeline, 9 s long, repeated until React removes the loader. The phase positions are exported as `AT` from `loader.ts`.

| start (s) | phase (caption) | what happens |
|-----------|-----------------|--------------|
| 0 | Reading the molecule | bonds draw in (`stroke-dashoffset` on `pathLength="1"`, stagger 0.035), atoms grow from radius 0 (`back.out`), O/N labels fade in |
| 1.1 | Passing messages between atoms | 3 rounds, 1.3 s apart: a pulse runs each way along every bond (0.9 s, `sine.inOut`); at the end of each round the atom-state glows grow brighter (radius 18 → 22 → 28, opacity 0.35 → 0.6 → 0.85) |
| 5 | Scoring each atom | glows fade, atoms bump 15 %, yellow Integrated Gradients halos grow to `15 + 22 × score` with opacity `0.35 + 0.65 × score` |
| 6.1 | Pooling and predicting | first arrow lights, vector cells fill column by column (0.6 s in total), second arrow lights, a warm scan sweeps each property head (stagger 0.12) |
| 8.4 | (still phase 4) | graph, vector and arrows fade out; the loop restarts at 9 s |

The property heads only show a scanning highlight, never a value, so nothing looks like a prediction before the data arrives.

**Phase bars and captions stay in sync by construction.** `showPhase(i, from, to)` adds the caption tween and the bar tween at the same timeline positions. The incoming pair starts 0.15 s after its phase begins, and the outgoing pair leaves in 0.15 s, so two captions never overlap. There is no second clock, CSS animation or timer that could drift.

SVG elements are animated through attributes only (`r`, `cx`, `cy`, opacity, dash offset), never transforms. This avoids GSAP measuring them with `getBBox`, which jsdom lacks.

### Clock, long waits and cleanup
- A `gsap.ticker` callback writes `<SMILES> · <seconds> s` into the clock, touching the DOM only when the text changes.
- After 12 s it sets `data-slow` on the host. The CSS then adds "Still working. The written interpretation can take a minute or two." under the clock. The backend gives the LLM up to 300 s (`llm_timeout_seconds`), so long waits are expected.
- The same callback is the cleanup check. React removes the host without notice, so when `host.isConnected` turns false, the loader kills its timeline, removes the ticker callback and deletes its own elements. `useAnalysisMotion` also disposes every loader it mounted when it unmounts.

### Reduced motion
The timeline is built but paused at 6 s, the attribution frame: graph drawn, halos shown, phase 3 ("Scoring each atom") visible. The clock and the slow-request line still update, since they are text, not motion.

### Styles (`MoleculeOverview.css`)
- `.analysis-loader`: a grid with the stage on top, the phase block bottom-left, and the status and clock bottom-right. It sets the landing atom palette (`--atom-c/o/n`).
- Graph: grey bonds with round caps; atoms filled from `--atom-*` with a `--surface` outline; lavender glows blurred 6px; pulses with a lavender drop shadow. The halos copy the real viewer's XAI marks (`--xai-highlight` at 16 % fill with a solid stroke).
- Readout: 18px vector cells whose fill alpha comes from a per-cell `--v`; head tracks with a `--warm` scan.
- Phase block: 32 × 3px bars (lit with `--accent`), captions stacked in one grid cell.
- ≤ 760px: only the graph is shown, centred, and the footer stacks into one column.

## Decorative 3D molecules
`styles/molecule3d.ts` is a small renderer with no framework dependency, used only for the background of the landing page. It is purely decorative: it is unrelated to the real 3D viewer in the overview card (planned for 3Dmol.js), and it is hidden from screen readers (`aria-hidden="true"` on `.molecule-field`).

### Data model

```ts
type Atom = { el: "C" | "O" | "N" | "H"; x: number; y: number; z: number }; // ångström
type Bond = [a: number, b: number, order: 1 | 2];                            // atom indices
type Molecule = {
    atoms: Atom[];
    bonds: Bond[];
    groups: { atoms: number[]; tone: "warm" | "accent" }[];                  // XAI highlight groups
};
```

The coordinates are approximate real geometry, built with small helpers:

| helper | builds |
|--------|--------|
| `ring` | six-membered ring with 1.4 Å radius |
| `hydrogen` | a hydrogen along a given direction |
| `methyl` | three tetrahedral hydrogens on an sp3 carbon |

| key | molecule | highlight groups |
|-----|----------|------------------|
| `aspirin` | acetylsalicylic acid, with the ester twisted out of plane | carboxylic acid (warm), acetyl ester (accent), ring (accent) |
| `caffeine` | purine core with 3 methyls | imidazole (accent), C6=O / N1-methyl (warm), C2=O / N3 (accent) |
| `salicylic` | salicylic acid | hydroxyl (accent), carboxylic acid (warm) |
| `benzene` | benzene | two halves of the ring |
| `ethanol` | ethanol | C–OH (warm), methyl carbon (accent) |
| `paracetamol` | paracetamol (acetaminophen), with the amide slightly twisted out of plane | hydroxyl (accent), amide N–C=O (warm), ring (accent) |
| `pyridine` | pyridine | N side of the ring (accent), far side (warm) |
| `glycine` | glycine | carboxylic acid (warm), amine + α-carbon (accent) |

**To add a molecule:**

1. Write a builder function that returns a `Molecule`.
2. Add it to `MOLECULES`.
3. Use its key in `data-molecule`.

### Rendering pipeline
`mountMolecule(container, molecule, { animate })` appends one `<svg class="molecule3d" viewBox="-1 -1 2 2">` and returns `{ dispose }`.

1. **Normalise.**
   - Centre the molecule on its centroid.
   - Scale it so the farthest atom, plus its radius, sits at 0.66. The model then fits the viewBox at any rotation.
   - Flip y, because SVG y points down.
2. **Build the DOM.**
   - `<defs>`: one radial gradient per element (a highlight at 30 %/26 % for a specular look, then the base colour, then a dark rim). The gradients use the `--atom-*` tokens through `color-mix`, so re-theming the tokens re-colours the molecules.
   - A blur filter for the highlight halo.
   - A `molecule3d__xai` group (halo ellipse, outline ellipse, pulse ring) behind a `molecule3d__scene` group.
   - In the scene: one `<circle>` per atom, and one `<g class="molecule3d__bond">` per bond, holding one `<line>` per bond order.
3. **Each frame** (`gsap.ticker`):
   - **Rotate.** About Y by the spin angle, then about X by `tilt + sin(t × wobble) × 0.35`.
   - **Project.** Perspective with camera distance `PERSPECTIVE = 3.2`, using the scale factor `f = 3.2 / (3.2 − z)`.
   - **Depth cues.** Radius is multiplied by `f`. Opacity ramps from 0.45 at the back to 1.0 at the front. Bonds use the average of their two atoms.
   - **Double bonds** are drawn as two parallel lines, offset perpendicular to the projected bond.
   - **Depth sort.** Atoms and bonds are sorted back-to-front; each bond sits just behind the nearer of its two atoms. Nodes are re-appended **only when the order changes**, to avoid needless DOM writes.
   - **Highlight ellipse.** It is fitted to the projected atoms of the active group by PCA:
     - the centre is the mean, and the rotation is the principal axis;
     - the radii are 1.7 × the standard deviation along each axis, plus padding (1.9 × the largest atom radius in the group);
     - so the ellipse follows the group as the molecule turns.

Per-molecule random parameters, so no two molecules turn alike:

| parameter | range |
|-----------|-------|
| starting angle | 0 – 2π |
| spin | 0.12 – 0.2 rad/s, random direction |
| X tilt | ±0.5 rad |
| wobble frequency | 0.18 – 0.3 |

The time step is capped at 50 ms, so a background tab does not cause a jump when it regains focus.

### Highlight cycle
A repeating GSAP timeline steps through `groups`. For each group:

1. it becomes the active group;
2. `level` goes from 0 to 1 over 0.9 s (`sine.inOut`);
3. it holds for 1.9 s;
4. `level` goes from 1 to 0 over 0.9 s.

So each group is shown for about 3.7 s. The cycle starts after a random 0.6–2.2 s delay. While a group is lit:

- its atoms grow 8 % and gain +0.3 opacity;
- the halo and outline ellipses fade in with `level`;
- a pulse ring expands from the group's first atom (1.2× → 2.8× its radius, fading out, repeating every 1.6 s).

`data-tone` on the ellipses and ring selects the colour in CSS:

| selector (in `AnalysisPage.css`) | style |
|----------------------------------|-------|
| `.molecule3d__bond line` | stroke `var(--bond)`, round caps |
| `.molecule3d__halo[data-tone=warm\|accent]` | 16–18 % warm or lavender fill (blurred) |
| `.molecule3d__outline[data-tone=…]`, `.molecule3d__pulse[data-tone=…]` | 75 % warm or lavender stroke |

### Slot placement
There are ten slots. Each one is placed by its **centre point, as a percentage of the viewport**, so the composition keeps its balance at any screen size. The slots form an even ring around the copy and the SMILES bar:

```
       far(2)      backdrop(9)
  near(1)                          near(5)
  mid(3)        [  copy + SMILES  ]   far(6)
  far(4)   backdrop(10)   far(8)      mid(7)
```

There are four molecules along the top, two at the sides and four along the bottom, five on each half of the screen. Bright near and mid molecules stay at the edges. Only faint far and backdrop molecules come towards the centre, so nothing competes with the text. Slots 9–10 carry the `molecule-field__item--backdrop` modifier.

Each slot is configured with custom properties:

| property | meaning |
|----------|---------|
| `--cx` / `--cy` | centre of the slot, as % of the viewport width / height |
| `--size` | box size; a `clamp(…, vw, …)`, so it scales with the screen |
| `--o` | depth opacity, applied to the `<svg>` so that GSAP is free to own the slot's own opacity |
| `--glow` | colour of a soft radial light behind the molecule (`::before`) |

Four depth layers are created by varying size, opacity, blur and tint (sizes are given at 1440px):

| slot | layer | molecule | size | opacity | centre (x, y) | look |
|------|-------|----------|------|---------|---------------|------|
| 1 | near | aspirin | 245px | 0.92 | 10 %, 25 % | lavender glow |
| 5 | near | caffeine | 230px | 0.88 | 89 %, 23 % | periwinkle glow |
| 3 | mid | paracetamol | 180px | 0.7 | 8 %, 61 % | warm glow |
| 7 | mid | salicylic | 187px | 0.68 | 86 %, 81 % | teal glow |
| 2 | far | benzene | 108px | 0.45 | 33 %, 18 % | blur 1.5px, periwinkle tint |
| 4 | far | glycine | 115px | 0.42 | 15 %, 87 % | blur 1.5px, periwinkle tint |
| 6 | far | ethanol | 101px | 0.38 | 94 %, 53 % | blur 1.5px, periwinkle tint |
| 8 | far | pyridine | 115px | 0.42 | 64 %, 88 % | blur 1.5px, periwinkle tint |
| 9 | backdrop | benzene | 374px | 0.14 | 64 %, 24 % | blur 2.5px, deep periwinkle monochrome |
| 10 | backdrop | caffeine | 317px | 0.12 | 37 %, 86 % | blur 2.5px, deep periwinkle monochrome |

The far and backdrop layers are tinted towards periwinkle (atmospheric perspective: distant objects take on the ambient colour), so they read as depth rather than as competing objects.

**Palette.** `.molecule-field` overrides the `--atom-*` tokens with a landing-only palette:

| element | colour |
|---------|--------|
| carbon | cool platinum `#c2c8da` |
| oxygen | rose `#f07890` |
| nitrogen | periwinkle `#8fa4ff` |
| hydrogen | pearl `#f3f1fa` |
| bonds | `--bond: #7d87a4` |

The far and backdrop layers override these again with their tints. The overrides are scoped to `.molecule-field`, so the real 3D viewer keeps the standard atom colours.

**Responsive:**

| width | visible slots |
|-------|---------------|
| ≤ 1280px | far layer hidden: near, mid and backdrop remain (6 molecules) |
| ≤ 1100px | mid layer hidden as well; the two near molecules shrink to 150px |
| ≤ 960px | whole field hidden; the gutters are too narrow to keep molecules clear of the copy |

Performance: each molecule updates about 20–40 SVG nodes per frame through attributes only. GSAP's ticker pauses automatically when the tab is hidden.