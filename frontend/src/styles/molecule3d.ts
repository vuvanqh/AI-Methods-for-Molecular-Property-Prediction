/**
 * Decorative 3D molecules for the landing page.
 *
 * A tiny framework-free renderer: atom coordinates (Å) are rotated every frame,
 * perspective-projected into an SVG and depth-sorted, so the ball-and-stick
 * models genuinely turn in 3D. An "XAI" highlight cycles through atom groups
 * to hint at attribution maps. Purely visual; mount into any empty element.
 */
import gsap from "gsap";

type AtomKind = "C" | "O" | "N" | "H";
type Atom = { el: AtomKind; x: number; y: number; z: number };
type Bond = [a: number, b: number, order: 1 | 2];
type Tone = "warm" | "accent";

export type Molecule = {
    atoms: Atom[];
    bonds: Bond[];
    /** atom groups the XAI highlight cycles through; the first atom anchors the pulse ring */
    groups: { atoms: number[]; tone: Tone }[];
};

export type MoleculeHandle = { dispose: () => void };

// ── Geometry ────────────────────────────────────────────────────────────────

function builder() {
    const atoms: Atom[] = [];
    const bonds: Bond[] = [];
    const add = (el: AtomKind, x: number, y: number, z = 0) => atoms.push({ el, x, y, z }) - 1;
    const bond = (a: number, b: number, order: 1 | 2 = 1) => void bonds.push([a, b, order]);

    /** hydrogen on atom `a` pointing along (ux, uy, uz) */
    const hydrogen = (a: number, ux: number, uy: number, uz = 0, len = 1.05) => {
        const n = Math.hypot(ux, uy, uz);
        const { x, y, z } = atoms[a];
        bond(a, add("H", x + (ux / n) * len, y + (uy / n) * len, z + (uz / n) * len));
    };

    /** three tetrahedral hydrogens on an sp3 carbon whose heavy-atom bond points opposite (ux, uy) */
    const methyl = (c: number, ux: number, uy: number) => {
        const n = Math.hypot(ux, uy);
        const [u0, u1] = [ux / n, uy / n];
        const [p0, p1] = [-u1, u0];
        hydrogen(c, 0.334 * u0 + 0.943 * p0, 0.334 * u1 + 0.943 * p1, 0);
        hydrogen(c, 0.334 * u0 - 0.471 * p0, 0.334 * u1 - 0.471 * p1, 0.816);
        hydrogen(c, 0.334 * u0 - 0.471 * p0, 0.334 * u1 - 0.471 * p1, -0.816);
    };

    /** six-membered ring, vertices from `start` degrees going clockwise */
    const ring = (el: AtomKind[], start = 90, r = 1.4) =>
        el.map((e, i) => {
            const a = ((start - i * 60) * Math.PI) / 180;
            return add(e, Math.cos(a) * r, Math.sin(a) * r);
        });

    const outward = (a: number) => {
        const { x, y } = atoms[a];
        return [x, y] as const;
    };

    return { atoms, bonds, add, bond, hydrogen, methyl, ring, outward };
}

function benzeneRing(b: ReturnType<typeof builder>) {
    const r = b.ring(["C", "C", "C", "C", "C", "C"]);
    r.forEach((a, i) => b.bond(a, r[(i + 1) % 6], i % 2 === 0 ? 2 : 1));
    return r;
}

function aspirin(): Molecule {
    const b = builder();
    const r = benzeneRing(b);
    [1, 2, 3, 4].forEach((i) => b.hydrogen(r[i], ...b.outward(r[i])));
    // carboxylic acid
    const c7 = b.add("C", 0, 2.9);
    const o8 = b.add("O", 1.1, 3.6);
    const o9 = b.add("O", -1.15, 3.55);
    b.bond(r[0], c7);
    b.bond(c7, o8, 2);
    b.bond(c7, o9);
    b.hydrogen(o9, 0.2, 1, 0.1, 0.96);
    // acetyl ester, twisted out of the ring plane
    const o10 = b.add("O", -2.45, 1.45, 0.25);
    const c11 = b.add("C", -3.55, 0.75, 0.75);
    const o12 = b.add("O", -3.55, -0.45, 1.05);
    const c13 = b.add("C", -4.8, 1.55, 0.95);
    b.bond(r[5], o10);
    b.bond(o10, c11);
    b.bond(c11, o12, 2);
    b.bond(c11, c13);
    b.methyl(c13, -0.8, 0.6);
    return {
        atoms: b.atoms,
        bonds: b.bonds,
        groups: [
            { atoms: [c7, o8, o9], tone: "warm" },
            { atoms: [o10, c11, o12, c13], tone: "accent" },
            { atoms: r, tone: "accent" },
        ],
    };
}

function salicylicAcid(): Molecule {
    const b = builder();
    const r = benzeneRing(b);
    [1, 2, 3, 4].forEach((i) => b.hydrogen(r[i], ...b.outward(r[i])));
    const c7 = b.add("C", 0, 2.9);
    const o8 = b.add("O", 1.1, 3.6);
    const o9 = b.add("O", -1.15, 3.55);
    b.bond(r[0], c7);
    b.bond(c7, o8, 2);
    b.bond(c7, o9);
    b.hydrogen(o9, 0.2, 1, 0, 0.96);
    const o10 = b.add("O", -2.45, 1.45);
    b.bond(r[5], o10);
    b.hydrogen(o10, -0.8, -0.6, 0, 0.96);
    return {
        atoms: b.atoms,
        bonds: b.bonds,
        groups: [
            { atoms: [o10, r[5]], tone: "accent" },
            { atoms: [c7, o8, o9], tone: "warm" },
        ],
    };
}

function benzene(): Molecule {
    const b = builder();
    const r = benzeneRing(b);
    r.forEach((a) => b.hydrogen(a, ...b.outward(a), 0, 1.08));
    return {
        atoms: b.atoms,
        bonds: b.bonds,
        groups: [
            { atoms: [r[0], r[1], r[2]], tone: "warm" },
            { atoms: [r[3], r[4], r[5]], tone: "accent" },
        ],
    };
}

function caffeine(): Molecule {
    const b = builder();
    // purine core: 6-ring N1 C2 N3 C4 C5 C6, imidazole fused on C4–C5
    const [n1, c2, n3, c4, c5, c6] = b.ring(["N", "C", "N", "C", "C", "C"], 90);
    const n7 = b.add("N", 2.252, -1.637);
    const c8 = b.add("C", 1.683, -2.916);
    const n9 = b.add("N", 0.291, -2.77);
    b.bond(n1, c2);
    b.bond(c2, n3);
    b.bond(n3, c4);
    b.bond(c4, c5, 2);
    b.bond(c5, c6);
    b.bond(c6, n1);
    b.bond(c5, n7);
    b.bond(n7, c8);
    b.bond(c8, n9, 2);
    b.bond(n9, c4);
    b.hydrogen(c8, 0.5, -0.866, 0, 1.08);
    const o2 = b.add("O", -2.29, 1.325);
    const o6 = b.add("O", 2.29, 1.325);
    b.bond(c2, o2, 2);
    b.bond(c6, o6, 2);
    const me1 = b.add("C", 0, 2.87);
    const me3 = b.add("C", -2.47, -1.43);
    const me7 = b.add("C", 3.67, -1.335);
    b.bond(n1, me1);
    b.bond(n3, me3);
    b.bond(n7, me7);
    b.methyl(me1, 0, 1);
    b.methyl(me3, -0.866, -0.5);
    b.methyl(me7, 0.978, 0.208);
    return {
        atoms: b.atoms,
        bonds: b.bonds,
        groups: [
            { atoms: [n7, c4, c5, c8, n9], tone: "accent" },
            { atoms: [o6, c6, n1, me1], tone: "warm" },
            { atoms: [o2, c2, n3], tone: "accent" },
        ],
    };
}

function ethanol(): Molecule {
    const b = builder();
    const c1 = b.add("C", 0, 0);
    const c2 = b.add("C", 1.52, 0);
    const o = b.add("O", 2.05, 1.4);
    b.bond(c1, c2);
    b.bond(c2, o);
    b.hydrogen(o, 1, -0.05, 0.1, 0.96);
    b.hydrogen(c2, 0.35, -0.5, 0.8);
    b.hydrogen(c2, 0.35, -0.5, -0.8);
    b.methyl(c1, -1, 0);
    return {
        atoms: b.atoms,
        bonds: b.bonds,
        groups: [
            { atoms: [o, c2], tone: "warm" },
            { atoms: [c1], tone: "accent" },
        ],
    };
}

function paracetamol(): Molecule {
    const b = builder();
    const r = benzeneRing(b);
    [1, 2, 4, 5].forEach((i) => b.hydrogen(r[i], ...b.outward(r[i])));
    // para-hydroxyl
    const o = b.add("O", 0, 2.76);
    b.bond(r[0], o);
    b.hydrogen(o, 0.9, 0.45, 0, 0.96);
    // acetamide, slightly twisted out of the ring plane
    const n = b.add("N", 0, -2.8, 0.15);
    const c = b.add("C", 1.2, -3.5, 0.35);
    const oc = b.add("O", 2.3, -2.95, 0.4);
    const me = b.add("C", 1.15, -5.0, 0.55);
    b.bond(r[3], n);
    b.hydrogen(n, -0.87, -0.3, -0.2, 1.01);
    b.bond(n, c);
    b.bond(c, oc, 2);
    b.bond(c, me);
    b.methyl(me, 0, -1);
    return {
        atoms: b.atoms,
        bonds: b.bonds,
        groups: [
            { atoms: [o, r[0]], tone: "accent" },
            { atoms: [n, c, oc], tone: "warm" },
            { atoms: r, tone: "accent" },
        ],
    };
}

function pyridine(): Molecule {
    const b = builder();
    const r = b.ring(["N", "C", "C", "C", "C", "C"]);
    r.forEach((a, i) => b.bond(a, r[(i + 1) % 6], i % 2 === 0 ? 2 : 1));
    r.slice(1).forEach((a) => b.hydrogen(a, ...b.outward(a), 0, 1.08));
    return {
        atoms: b.atoms,
        bonds: b.bonds,
        groups: [
            { atoms: [r[0], r[1], r[5]], tone: "accent" },
            { atoms: [r[2], r[3], r[4]], tone: "warm" },
        ],
    };
}

function glycine(): Molecule {
    const b = builder();
    const n = b.add("N", 0, 0);
    const ca = b.add("C", 1.46, 0.1);
    const c = b.add("C", 2.1, 1.45);
    const o1 = b.add("O", 1.45, 2.45);
    const o2 = b.add("O", 3.42, 1.5);
    b.bond(n, ca);
    b.bond(ca, c);
    b.bond(c, o1, 2);
    b.bond(c, o2);
    b.hydrogen(o2, 0.6, 0.8, 0, 0.96);
    b.hydrogen(n, -0.5, 0.8, 0.3, 1.01);
    b.hydrogen(n, -0.5, -0.6, -0.6, 1.01);
    b.hydrogen(ca, 0.2, -0.6, 0.85);
    b.hydrogen(ca, 0.2, -0.6, -0.85);
    return {
        atoms: b.atoms,
        bonds: b.bonds,
        groups: [
            { atoms: [c, o1, o2], tone: "warm" },
            { atoms: [n, ca], tone: "accent" },
        ],
    };
}

export const MOLECULES: Record<string, Molecule> = {
    aspirin: aspirin(),
    caffeine: caffeine(),
    benzene: benzene(),
    ethanol: ethanol(),
    salicylic: salicylicAcid(),
    paracetamol: paracetamol(),
    pyridine: pyridine(),
    glycine: glycine(),
};

// ── Rendering ───────────────────────────────────────────────────────────────

const NS = "http://www.w3.org/2000/svg";
const RADIUS: Record<AtomKind, number> = { C: 0.42, N: 0.4, O: 0.4, H: 0.26 };
const ATOM_VAR: Record<AtomKind, string> = { C: "--atom-c", N: "--atom-n", O: "--atom-o", H: "--atom-h" };
const PERSPECTIVE = 3.2; // camera distance in normalised units (molecule fits in ~0.66)
let uid = 0;

function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    return node;
}

type Drawable = { node: SVGElement; depth: number };

export function mountMolecule(
    container: HTMLElement,
    molecule: Molecule,
    { animate = true }: { animate?: boolean } = {},
): MoleculeHandle {
    const id = `mol${++uid}`;

    // centre on the centroid and normalise so the model fits the viewBox at any rotation
    const n = molecule.atoms.length;
    const cx = molecule.atoms.reduce((s, a) => s + a.x, 0) / n;
    const cy = molecule.atoms.reduce((s, a) => s + a.y, 0) / n;
    const cz = molecule.atoms.reduce((s, a) => s + a.z, 0) / n;
    const maxR = Math.max(...molecule.atoms.map((a) => Math.hypot(a.x - cx, a.y - cy, a.z - cz) + RADIUS[a.el]));
    const scale = 0.66 / maxR;
    const pts = molecule.atoms.map((a) => ({
        el: a.el,
        x: (a.x - cx) * scale,
        y: -(a.y - cy) * scale, // SVG y points down
        z: (a.z - cz) * scale,
        r: RADIUS[a.el] * scale,
    }));
    const depthRange = 0.66;
    const bondWidth = 0.13 * scale;

    // ── DOM ──
    const root = svg("svg", { viewBox: "-1 -1 2 2", class: "molecule3d" });
    const defs = svg("defs");
    (["C", "N", "O", "H"] as AtomKind[]).forEach((el) => {
        const g = svg("radialGradient", { id: `${id}-${el}`, cx: "35%", cy: "30%", r: "75%", fx: "32%", fy: "26%" });
        const base = `var(${ATOM_VAR[el]})`;
        // specular dot → soft highlight → body → deep rim: reads as polished, not flat
        [
            ["0%", `color-mix(in srgb, ${base} 22%, white)`],
            ["18%", `color-mix(in srgb, ${base} 72%, white)`],
            ["52%", base],
            ["100%", `color-mix(in srgb, ${base} 32%, #050912)`],
        ].forEach(([offset, color]) => {
            const stop = svg("stop", { offset });
            stop.style.stopColor = color;
            g.append(stop);
        });
        defs.append(g);
    });
    const blur = svg("filter", { id: `${id}-halo`, x: "-50%", y: "-50%", width: "200%", height: "200%" });
    blur.append(svg("feGaussianBlur", { stdDeviation: 0.05 }));
    defs.append(blur);
    root.append(defs);

    const xai = svg("g", { class: "molecule3d__xai" });
    const halo = svg("ellipse", { class: "molecule3d__halo", filter: `url(#${id}-halo)` });
    const outline = svg("ellipse", { class: "molecule3d__outline", "stroke-width": 0.012 });
    const pulse = svg("circle", { class: "molecule3d__pulse", "stroke-width": 0.01 });
    xai.append(halo, outline, pulse);
    const scene = svg("g", { class: "molecule3d__scene" });
    root.append(xai, scene);

    const atomNodes = pts.map((p) => {
        const c = svg("circle", { fill: `url(#${id}-${p.el})` });
        scene.append(c);
        return { node: c as SVGElement, depth: 0 } satisfies Drawable;
    });
    const bondNodes = molecule.bonds.map(([, , order]) => {
        const g = svg("g", { class: "molecule3d__bond" });
        const lines = Array.from({ length: order }, () => svg("line"));
        g.append(...lines);
        scene.append(g);
        return { node: g as SVGElement, depth: 0, lines };
    });
    container.append(root);

    // ── state ──
    let ry = gsap.utils.random(0, Math.PI * 2);
    let t = gsap.utils.random(0, 100);
    const spin = gsap.utils.random(0.12, 0.2) * (Math.random() < 0.5 ? -1 : 1);
    const tilt = gsap.utils.random(-0.5, 0.5);
    const wobble = gsap.utils.random(0.18, 0.3);
    const xaiState = { group: 0, level: animate ? 0 : 1, pulse: 0 };
    let order: Drawable[] = [];

    const projected = pts.map(() => ({ x: 0, y: 0, z: 0, f: 1 }));

    function frame() {
        const rx = tilt + Math.sin(t * wobble) * 0.35;
        const [sy, cyR, sx, cxR] = [Math.sin(ry), Math.cos(ry), Math.sin(rx), Math.cos(rx)];
        pts.forEach((p, i) => {
            // rotate about Y, then X
            const x1 = p.x * cyR + p.z * sy;
            const z1 = -p.x * sy + p.z * cyR;
            const y2 = p.y * cxR - z1 * sx;
            const z2 = p.y * sx + z1 * cxR;
            const f = PERSPECTIVE / (PERSPECTIVE - z2);
            const o = projected[i];
            o.x = x1 * f;
            o.y = y2 * f;
            o.z = z2;
            o.f = f;
        });

        const group = molecule.groups[xaiState.group];
        const inGroup = new Set(group?.atoms);
        const fade = (z: number) => 0.45 + 0.55 * ((z + depthRange) / (2 * depthRange));

        pts.forEach((p, i) => {
            const o = projected[i];
            const lit = inGroup.has(i) ? xaiState.level : 0;
            const node = atomNodes[i];
            node.node.setAttribute("cx", o.x.toFixed(4));
            node.node.setAttribute("cy", o.y.toFixed(4));
            node.node.setAttribute("r", (p.r * o.f * (1 + 0.08 * lit)).toFixed(4));
            node.node.setAttribute("opacity", Math.min(1, fade(o.z) + 0.3 * lit).toFixed(3));
            node.depth = o.z;
        });

        molecule.bonds.forEach(([a, b], i) => {
            const pa = projected[a];
            const pb = projected[b];
            const entry = bondNodes[i];
            const f = (pa.f + pb.f) / 2;
            const w = bondWidth * f;
            const dx = pb.x - pa.x;
            const dy = pb.y - pa.y;
            const len = Math.hypot(dx, dy) || 1;
            const off = entry.lines.length === 2 ? w * 0.9 : 0;
            entry.lines.forEach((line, k) => {
                const s = entry.lines.length === 2 ? (k === 0 ? 1 : -1) : 0;
                const ox = (-dy / len) * off * s;
                const oy = (dx / len) * off * s;
                line.setAttribute("x1", (pa.x + ox).toFixed(4));
                line.setAttribute("y1", (pa.y + oy).toFixed(4));
                line.setAttribute("x2", (pb.x + ox).toFixed(4));
                line.setAttribute("y2", (pb.y + oy).toFixed(4));
                line.setAttribute("stroke-width", (entry.lines.length === 2 ? w * 0.6 : w).toFixed(4));
            });
            entry.node.setAttribute("opacity", fade((pa.z + pb.z) / 2).toFixed(3));
            // bonds sit just behind the nearer of their two atoms
            entry.depth = Math.min(pa.z, pb.z) - 0.001;
        });

        // depth sort; only touch the DOM when the order actually changes
        const next = [...atomNodes, ...bondNodes].sort((p, q) => p.depth - q.depth);
        if (next.some((d, i) => d !== order[i])) {
            next.forEach((d) => scene.append(d.node));
            order = next;
        }

        // XAI attribution ellipse fitted to the projected group
        if (group && xaiState.level > 0.001) {
            const g = group.atoms.map((i) => projected[i]);
            const mx = g.reduce((s, p) => s + p.x, 0) / g.length;
            const my = g.reduce((s, p) => s + p.y, 0) / g.length;
            let sxx = 0;
            let syy = 0;
            let sxy = 0;
            g.forEach((p) => {
                sxx += (p.x - mx) ** 2;
                syy += (p.y - my) ** 2;
                sxy += (p.x - mx) * (p.y - my);
            });
            [sxx, syy, sxy] = [sxx / g.length, syy / g.length, sxy / g.length];
            const mid = (sxx + syy) / 2;
            const spread = Math.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2);
            const angle = (0.5 * Math.atan2(2 * sxy, sxx - syy) * 180) / Math.PI;
            const pad = Math.max(...group.atoms.map((i) => pts[i].r * projected[i].f)) * 1.9;
            const rx = Math.sqrt(mid + spread) * 1.7 + pad;
            const ry2 = Math.sqrt(Math.max(mid - spread, 0)) * 1.7 + pad;
            const tone = group.tone;
            for (const el of [halo, outline]) {
                el.setAttribute("cx", mx.toFixed(4));
                el.setAttribute("cy", my.toFixed(4));
                el.setAttribute("rx", rx.toFixed(4));
                el.setAttribute("ry", ry2.toFixed(4));
                el.setAttribute("transform", `rotate(${angle.toFixed(2)} ${mx.toFixed(4)} ${my.toFixed(4)})`);
                el.setAttribute("data-tone", tone);
            }
            const anchor = projected[group.atoms[0]];
            const anchorR = pts[group.atoms[0]].r * anchor.f;
            pulse.setAttribute("cx", anchor.x.toFixed(4));
            pulse.setAttribute("cy", anchor.y.toFixed(4));
            pulse.setAttribute("r", (anchorR * (1.2 + 1.6 * xaiState.pulse)).toFixed(4));
            pulse.setAttribute("data-tone", tone);
            pulse.setAttribute("opacity", ((1 - xaiState.pulse) * xaiState.level * 0.7).toFixed(3));
            xai.setAttribute("opacity", xaiState.level.toFixed(3));
        } else {
            xai.setAttribute("opacity", "0");
        }
    }

    frame();
    if (!animate) return { dispose: () => root.remove() };

    const tick = (_time: number, deltaTime: number) => {
        const dt = Math.min(deltaTime, 50) / 1000;
        ry += spin * dt;
        t += dt;
        frame();
    };
    gsap.ticker.add(tick);

    // cycle the attribution highlight through the groups
    const cycle = gsap.timeline({ repeat: -1, delay: gsap.utils.random(0.6, 2.2) });
    molecule.groups.forEach((_, i) => {
        cycle
            .call(() => void (xaiState.group = i))
            .to(xaiState, { level: 1, duration: 0.9, ease: "sine.inOut" })
            .to(xaiState, { level: 0, duration: 0.9, ease: "sine.inOut" }, "+=1.9");
    });
    const ring = gsap.to(xaiState, { pulse: 1, duration: 1.6, ease: "power1.out", repeat: -1 });

    return {
        dispose() {
            gsap.ticker.remove(tick);
            cycle.kill();
            ring.kill();
            root.remove();
        },
    };
}
