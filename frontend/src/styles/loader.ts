/**
 * Loading animation for the analysis request: a 2D sketch of the GNN pipeline, left to right.
 *
 * A molecular graph draws in, message-passing pulses run along every bond for three rounds,
 * Integrated Gradients halos mark the atoms, then the atoms pool into a molecule vector that feeds
 * the property heads. One GSAP timeline drives all of it, the phase bars and captions included, so
 * each bar and its caption change at the same timeline position and cannot drift apart.
 *
 * Decorative only (aria-hidden): the host keeps its own status text for screen readers.
 * Mount into the loader element; it removes itself once React drops that element.
 */
import gsap from "gsap";
import { MOLECULES, svg, type Molecule } from "./molecule3d";

export type LoaderHandle = { dispose: () => void; timeline: gsap.core.Timeline };

const W = 400;
const H = 290;
const PAD = 34;
const SLOW_AFTER = 12; // seconds before the "still working" line shows
const SKETCHABLE = ["aspirin", "paracetamol", "salicylic"]; // read well flat; the others overlap in 2D
const HEADS = ["CYP1A2", "CYP2C9", "hERG", "NR-AR", "SR-ARE", "BBBP"];
const PHASES = [
    ["Reading the molecule", "SMILES → atoms and bonds"],
    ["Passing messages between atoms", "Chemprop GNN · 3 rounds along every bond"],
    ["Scoring each atom", "Integrated Gradients attribution"],
    ["Pooling and predicting", "atom states → molecule vector → properties"],
];
/** timeline positions (s) of each phase; the loop is 9 s long */
export const AT = { read: 0, pass: 1.1, score: 5, pool: 6.1, fade: 8.4, end: 9 };
const STILL_AT = 6; // reduced motion: hold the attribution frame

function html<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text = "") {
    const node = document.createElement(tag);
    node.className = className;
    node.textContent = text;
    return node;
}

/** heavy atoms in viewBox coordinates, laid out landscape, each scored by the molecule's XAI groups */
function sketch(molecule: Molecule) {
    const keep = molecule.atoms.flatMap((a, i) => (a.el === "H" ? [] : [i]));
    const index = new Map(keep.map((atom, k) => [atom, k]));
    const score = (i: number) => Math.max(0.15, ...molecule.groups.map((g, k) => (g.atoms.includes(i) ? ([1, 0.65][k] ?? 0.4) : 0)));
    let pts = keep.map((i) => ({ el: molecule.atoms[i].el, x: molecule.atoms[i].x, y: molecule.atoms[i].y, s: score(i) }));
    const span = (k: "x" | "y") => Math.max(...pts.map((p) => p[k])) - Math.min(...pts.map((p) => p[k]));
    if (span("y") > span("x")) pts = pts.map((p) => ({ ...p, x: -p.y, y: p.x }));

    const [xs, ys] = [pts.map((p) => p.x), pts.map((p) => p.y)];
    const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const k = Math.min((W - 2 * PAD) / (maxX - minX), (H - 2 * PAD) / (maxY - minY), 56);
    const [ox, oy] = [(W - (maxX - minX) * k) / 2, (H - (maxY - minY) * k) / 2];
    const atoms = pts.map((p) => ({ ...p, x: ox + (p.x - minX) * k, y: oy + (maxY - p.y) * k, r: p.el === "C" ? 8 : 11 }));
    const bonds = molecule.bonds.flatMap(([a, b, order]) => (index.has(a) && index.has(b) ? [[index.get(a)!, index.get(b)!, order] as const] : []));
    return { atoms, bonds };
}

export function mountLoader(host: HTMLElement, { animate = true }: { animate?: boolean } = {}): LoaderHandle {
    const molecule = MOLECULES[host.dataset.molecule ?? ""] ?? MOLECULES[gsap.utils.random(SKETCHABLE)];
    const { atoms, bonds } = sketch(molecule);

    // ── graph ──
    const graph = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "analysis-loader__graph" });
    const layer = () => graph.appendChild(svg("g"));
    const [xaiLayer, msgLayer, bondLayer, pulseLayer, atomLayer] = [layer(), layer(), layer(), layer(), layer()];
    const xai = atoms.map((a) => xaiLayer.appendChild(svg("circle", { cx: a.x, cy: a.y, class: "analysis-loader__xai" })));
    const msg = atoms.map((a) => msgLayer.appendChild(svg("circle", { cx: a.x, cy: a.y, class: "analysis-loader__msg" })));
    const lines = bonds.flatMap(([a, b, order]) => {
        const [p, q] = [atoms[a], atoms[b]];
        const len = Math.hypot(q.x - p.x, q.y - p.y);
        const [nx, ny] = [(-(q.y - p.y) / len) * 3, ((q.x - p.x) / len) * 3];
        return (order === 2 ? [1, -1] : [0]).map((s) =>
            bondLayer.appendChild(svg("line", { x1: p.x + nx * s, y1: p.y + ny * s, x2: q.x + nx * s, y2: q.y + ny * s, pathLength: 1 })),
        );
    });
    // one pulse each way per bond: messages flow in both directions
    const routes = bonds.flatMap(([a, b]) => [[atoms[a], atoms[b]], [atoms[b], atoms[a]]]);
    const pulses = routes.map(([from]) => pulseLayer.appendChild(svg("circle", { cx: from.x, cy: from.y, r: 3.2, class: "analysis-loader__pulse" })));
    const atomCircles = atoms.map((a) => {
        const g = atomLayer.appendChild(svg("g", { class: "analysis-loader__atom", "data-el": a.el }));
        return g.appendChild(svg("circle", { cx: a.x, cy: a.y, r: a.r }));
    });
    const labels = atoms.flatMap((a, i) => {
        if (a.el === "C") return [];
        const text = svg("text", { x: a.x, y: a.y, "text-anchor": "middle", "dominant-baseline": "central" });
        text.textContent = a.el;
        return [atomCircles[i].parentNode!.appendChild(text)];
    });

    // ── readout: molecule vector and property heads ──
    const flows = [html("span", "analysis-loader__flow"), html("span", "analysis-loader__flow")];
    const vector = html("div", "analysis-loader__vector");
    const cells = Array.from({ length: 32 }, () => {
        const slot = vector.appendChild(html("span", "analysis-loader__cell"));
        const fill = slot.appendChild(html("span", "analysis-loader__cell-fill"));
        fill.style.setProperty("--v", gsap.utils.random(0.15, 1).toFixed(2));
        return fill;
    });
    const heads = html("div", "analysis-loader__heads");
    const scans = HEADS.map((name) => {
        const row = heads.appendChild(html("div", "analysis-loader__head"));
        row.append(html("span", "", name));
        const track = row.appendChild(html("span", "analysis-loader__head-track"));
        return track.appendChild(html("span", "analysis-loader__head-scan"));
    });
    const readout = (title: string, body: HTMLElement) => {
        const block = html("div", "analysis-loader__readout");
        block.append(html("p", "analysis-loader__eyebrow", title), body);
        return block;
    };
    const stage = html("div", "analysis-loader__stage");
    stage.append(graph, flows[0], readout("Molecule vector", vector), flows[1], readout("Property heads", heads));

    // ── phase bars + captions, and the clock ──
    const phase = html("div", "analysis-loader__phase");
    const segs = phase.appendChild(html("div", "analysis-loader__segs"));
    const bars = PHASES.map(() => segs.appendChild(html("span", "analysis-loader__seg")).appendChild(html("span", "")));
    const captionBox = phase.appendChild(html("div", "analysis-loader__captions"));
    const captions = PHASES.map(([title, detail]) => {
        const p = captionBox.appendChild(html("p", "analysis-loader__caption"));
        p.append(html("strong", "", title), html("span", "", detail));
        return p;
    });
    const clock = html("span", "analysis-loader__clock");
    for (const node of [stage, phase, clock]) node.setAttribute("aria-hidden", "true");
    host.append(stage, phase, clock);

    // ── timeline (attr tweens only on SVG: no transforms, so no getBBox) ──
    gsap.set([...captions, ...bars, ...pulses], { autoAlpha: 0 });
    gsap.set(lines, { strokeDashoffset: 1 });
    gsap.set([...xai, ...msg], { attr: { r: 0 }, opacity: 0 });
    gsap.set(scans, { xPercent: -100 });
    const tl = gsap.timeline({ repeat: -1, paused: !animate });

    // a phase's bar and caption always tween together, at the same position;
    // the outgoing pair leaves first so two captions never overlap
    const showPhase = (i: number, from: number, to: number) =>
        tl.fromTo(captions[i], { autoAlpha: 0, y: 4 }, { autoAlpha: 1, y: 0, duration: 0.25 }, from + 0.15)
            .fromTo(bars[i], { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.25 }, from + 0.15)
            .to(captions[i], { autoAlpha: 0, y: -4, duration: 0.15 }, to)
            .to(bars[i], { autoAlpha: 0, duration: 0.15 }, to);
    showPhase(0, AT.read, AT.pass);
    showPhase(1, AT.pass, AT.score);
    showPhase(2, AT.score, AT.pool);
    showPhase(3, AT.pool, AT.end - 0.15);

    // reading: bonds draw, atoms pop in
    tl.to(lines, { strokeDashoffset: 0, duration: 0.7, stagger: 0.035, ease: "power2.out" }, AT.read)
        .fromTo(atomCircles, { attr: { r: 0 } }, { attr: { r: (i: number) => atoms[i].r }, duration: 0.5, stagger: 0.03, ease: "back.out(2)" }, AT.read + 0.1)
        .fromTo(labels, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.3 }, AT.read + 0.4);

    // message passing: 3 rounds, atom states glow brighter each round
    [0, 1, 2].forEach((round) => {
        const t = AT.pass + round * 1.3;
        tl.fromTo(
            pulses,
            { attr: { cx: (i: number) => routes[i][0].x, cy: (i: number) => routes[i][0].y }, autoAlpha: 1 },
            { attr: { cx: (i: number) => routes[i][1].x, cy: (i: number) => routes[i][1].y }, duration: 0.9, ease: "sine.inOut" },
            t,
        )
            .to(pulses, { autoAlpha: 0, duration: 0.15 }, t + 0.85)
            .to(msg, { attr: { r: [18, 22, 28][round] }, opacity: [0.35, 0.6, 0.85][round], duration: 0.35, ease: "power2.out" }, t + 0.8);
    });

    // attribution: the atom states give way to Integrated Gradients halos sized by score
    tl.to(msg, { attr: { r: 34 }, opacity: 0, duration: 0.5 }, AT.score)
        .to(atomCircles, { attr: { r: (i: number) => atoms[i].r * 1.15 }, duration: 0.2, yoyo: true, repeat: 1, stagger: 0.02 }, AT.score)
        .to(xai, { attr: { r: (i: number) => 15 + 22 * atoms[i].s }, opacity: (i: number) => 0.35 + 0.65 * atoms[i].s, duration: 0.6, stagger: 0.02, ease: "power3.out" }, AT.score + 0.1);

    // readout: atoms pool into the molecule vector, the property heads read it
    tl.to(flows[0], { opacity: 1, duration: 0.3 }, AT.pool)
        .fromTo(cells, { autoAlpha: 0, scale: 0.3 }, { autoAlpha: 1, scale: 1, duration: 0.4, ease: "power3.out", stagger: { amount: 0.6, grid: [4, 8], axis: "x" } }, AT.pool + 0.1)
        .to(flows[1], { opacity: 1, duration: 0.3 }, AT.pool + 0.6)
        .fromTo(scans, { xPercent: -100 }, { xPercent: 250, duration: 1.1, ease: "sine.inOut", stagger: 0.12 }, AT.pool + 0.7)
        .to([graph, vector, ...flows], { autoAlpha: 0, duration: 0.5, ease: "power1.in" }, AT.fade);

    if (!animate) tl.seek(STILL_AT);

    // the clock doubles as the cleanup check: React removes the host without telling us
    const start = performance.now();
    const prefix = host.dataset.smiles ? `${host.dataset.smiles} · ` : "";
    const tick = () => {
        if (!host.isConnected) return dispose();
        const seconds = (performance.now() - start) / 1000;
        const text = `${prefix}${seconds.toFixed(1)} s`;
        if (clock.textContent !== text) clock.textContent = text;
        if (seconds > SLOW_AFTER) host.dataset.slow = "";
    };
    function dispose() {
        gsap.ticker.remove(tick);
        tl.kill();
        stage.remove();
        phase.remove();
        clock.remove();
    }
    gsap.ticker.add(tick);
    tick();

    return { dispose, timeline: tl };
}
