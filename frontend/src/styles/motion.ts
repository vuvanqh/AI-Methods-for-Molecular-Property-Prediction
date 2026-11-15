/**
 * All page motion lives here (GSAP). Components only carry class names and
 * `data-animate` attributes; this module finds them.
 *
 * Usage (AnalysisPage):
 *
 *   const { exitLanding, exitResults } = useAnalysisMotion({ analysisStarted })
 *
 *   async function analyze(smiles) { await exitLanding(); navigate(`/${encodeURIComponent(smiles)}`) }
 *   async function newAnalysis()   { await exitResults(); navigate("/") }
 *
 *   `scope` (a ref) is optional and defaults to the document. `analysisStarted` must become true
 *   in the same render that mounts the results (deriving it from the route guarantees that).
 *
 * Hooks into the markup:
 *   .navbar                         fades down on first load
 *   .landing-hero > *               landing copy, rises in / blurs out
 *   .molecule-field__item           background 3D molecule slot (mounted once, shown on every view),
 *                                   `data-molecule` = a key of MOLECULES in molecule3d.ts
 *   .smiles-input-container         glides between landing and results position (Flip)
 *   .smiles-examples__label/__item  landing example tiles: rise in after the bar, name first then its
 *                                   .smiles-example__smiles; leave together with the landing copy
 *   .analysis-hero > *              results heading
 *   .analysis-results .card         result cards, staggered
 *   [data-animate="stagger"]        list whose children stagger in (also when it mounts later, e.g. after loading)
 *   [data-animate="count"]          number counts up to `data-value`
 *   .property-family-select         changing it re-staggers the property rows
 *   [data-animate="loader"]         analysis loading animation (loader.ts), mounted whenever it appears,
 *                                   also with reduced motion (as a still frame); `data-smiles` shows in its clock
 */
import gsap from "gsap";
import { Flip } from "gsap/Flip";
import { useGSAP } from "@gsap/react";
import { useEffect, useRef, type RefObject } from "react";
import { MOLECULES, mountMolecule } from "./molecule3d";
import { mountLoader, type LoaderHandle } from "./loader";

gsap.registerPlugin(useGSAP, Flip);

const SEL = {
    nav: ".navbar",
    landingText: ".landing-hero > *",
    molecules: ".molecule-field__item",
    smilesBar: ".smiles-input-container",
    examples: ".smiles-examples__label, .smiles-examples__item",
    resultsText: ".analysis-hero > *",
    cards: ".analysis-results .card",
    lists: '[data-animate="stagger"]',
    counts: '[data-animate="count"]',
    loader: '[data-animate="loader"]',
};

const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Animate the numeric text node inside `el` from 0 to its `data-value`, leaving React's node in place. */
function countUp(el: HTMLElement, delay = 0) {
    const raw = el.dataset.value ?? "";
    const target = Number(raw);
    const node = [...el.childNodes].find(
        (n) => n.nodeType === Node.TEXT_NODE && n.nodeValue?.trim() !== "" && !Number.isNaN(Number(n.nodeValue)),
    );
    if (!node || !Number.isFinite(target)) return;
    const final = node.nodeValue;
    const decimals = (raw.split(".")[1] ?? "").length;
    const proxy = { v: 0 };
    node.nodeValue = (0).toFixed(decimals);
    gsap.to(proxy, {
        v: target,
        duration: 1.1,
        delay,
        ease: "power2.out",
        onUpdate: () => void (node.nodeValue = proxy.v.toFixed(decimals)),
        onComplete: () => void (node.nodeValue = final),
    });
}

function revealList(list: Element, delay = 0) {
    gsap.fromTo(
        list.children,
        { y: 8, autoAlpha: 0 },
        { y: 0, autoAlpha: 1, duration: 0.5, delay, stagger: 0.04, ease: "power2.out", clearProps: "transform" },
    );
    list.querySelectorAll<HTMLElement>(SEL.counts).forEach((el, i) => countUp(el, delay + 0.1 + i * 0.04));
}

/** Example tiles rise in one after another; inside each, the name lands before its SMILES slides in. */
function revealExamples(targets: HTMLElement[], delay: number) {
    if (!targets.length) return;
    const tl = gsap.timeline({ delay });
    targets.forEach((el, i) => {
        const at = i * 0.08;
        tl.fromTo(el, { y: 14, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: 0.6, ease: "power3.out", clearProps: "transform" }, at);
        const name = el.querySelector(".smiles-example__name");
        const smiles = el.querySelector(".smiles-example__smiles");
        if (name) tl.fromTo(name, { y: 6, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: 0.45, ease: "power2.out", clearProps: "transform" }, at + 0.1);
        if (smiles) tl.fromTo(smiles, { x: -6, autoAlpha: 0 }, { x: 0, autoAlpha: 1, duration: 0.5, ease: "power2.out", clearProps: "transform" }, at + 0.35);
    });
}

type Options = { analysisStarted: boolean; scope?: RefObject<HTMLElement | null> };

export function useAnalysisMotion({ analysisStarted, scope }: Options) {
    const flipState = useRef<Flip.FlipState | null>(null);
    const scopeRoot = () => scope?.current ?? document;

    // background molecules + navbar: mounted once and kept alive on both landing and results
    useGSAP(
        () => {
            const reduced = prefersReducedMotion();
            const q = gsap.utils.selector(scopeRoot());
            const items = q(SEL.molecules) as HTMLElement[];
            const handles = items.flatMap((el) => {
                const mol = MOLECULES[el.dataset.molecule ?? ""];
                return mol ? [mountMolecule(el, mol, { animate: !reduced })] : [];
            });

            if (reduced) {
                gsap.fromTo(items, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.2 });
            } else {
                gsap.fromTo(q(SEL.nav), { y: -8, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: 0.6, ease: "power3.out", clearProps: "transform" });
                gsap.fromTo(
                    items,
                    { autoAlpha: 0, scale: 0.9, y: () => gsap.utils.random(-14, 14) },
                    { autoAlpha: 1, scale: 1, y: 0, duration: 1.4, stagger: 0.12, delay: 0.1, ease: "power2.out" },
                );
                // gentle drift on top of the 3D rotation
                items.forEach((el) =>
                    gsap.to(el, {
                        y: gsap.utils.random(14, 22) * (Math.random() < 0.5 ? -1 : 1),
                        x: gsap.utils.random(-6, 6),
                        duration: gsap.utils.random(5, 9),
                        delay: 1.5 + gsap.utils.random(0, 1.5),
                        ease: "sine.inOut",
                        yoyo: true,
                        repeat: -1,
                    }),
                );
            }
            return () => handles.forEach((h) => h.dispose());
        },
        { scope },
    );

    // landing ⇄ results: copy rises in, SMILES bar glides between its two positions
    const { contextSafe } = useGSAP(
        () => {
            const reduced = prefersReducedMotion();
            const q = gsap.utils.selector(scopeRoot());
            const bar = q(SEL.smilesBar);
            const text = q(analysisStarted ? SEL.resultsText : SEL.landingText);
            const cards = analysisStarted ? q(SEL.cards) : [];
            const examples = analysisStarted ? [] : q(SEL.examples);
            const glide = flipState.current;
            flipState.current = null;

            if (reduced) {
                const all = [...text, ...cards, ...examples];
                if (all.length) gsap.fromTo(all, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.2 });
                return;
            }
            // after the bar: its first-load entrance, or its glide back from the results
            revealExamples(examples, glide ? 0.45 : 0.95);
            if (glide) Flip.from(glide, { targets: bar, duration: 0.7, ease: "power3.inOut", scale: false });
            else if (!analysisStarted) gsap.fromTo(bar, { y: 16, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: 0.8, delay: 0.6, ease: "power3.out", clearProps: "transform" });

            if (text.length) gsap.fromTo(
                text,
                { y: 18, autoAlpha: 0, filter: "blur(6px)" },
                { y: 0, autoAlpha: 1, filter: "blur(0px)", duration: 0.9, stagger: 0.09, delay: analysisStarted ? 0.05 : 0.25, ease: "power3.out", clearProps: "transform,filter" },
            );
            // cards present at this point (the loading placeholder); cards that mount later fade in via CSS
            if (cards.length) gsap.fromTo(cards, { y: 24, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: 0.8, stagger: 0.08, delay: 0.15, ease: "power3.out", clearProps: "transform" });
        },
        { scope, dependencies: [analysisStarted] },
    );

    useEffect(() => {
        const reduced = prefersReducedMotion();
        const within = (node: HTMLElement, sel: string) => (node.matches(sel) ? [node] : [...node.querySelectorAll<HTMLElement>(sel)]);
        const loaders: LoaderHandle[] = [];
        const mount = (el: HTMLElement) => void loaders.push(mountLoader(el, { animate: !reduced }));
        // the request can already be pending when this mounts (a direct link to /:smiles)
        document.querySelectorAll<HTMLElement>(SEL.loader).forEach(mount);

        // re-stagger the property rows when the family selector changes (React re-renders first)
        const onChange = (e: Event) => {
            if (!(e.target instanceof Element) || !e.target.matches(".property-family-select")) return;
            const card = e.target.closest(".card");
            requestAnimationFrame(() => card?.querySelectorAll(SEL.lists).forEach((list) => revealList(list)));
        };
        // nodes that mount later: the loader when a request starts, and lists that stagger in and count up when it resolves
        const observer = new MutationObserver((mutations) => {
            for (const m of mutations) {
                for (const node of m.addedNodes) {
                    if (!(node instanceof HTMLElement)) continue; // skips the molecules' SVG reordering
                    within(node, SEL.loader).forEach(mount);
                    if (!reduced) within(node, SEL.lists).forEach((list) => revealList(list, 0.2));
                }
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
        if (!reduced) document.addEventListener("change", onChange);
        return () => {
            observer.disconnect();
            document.removeEventListener("change", onChange);
            loaders.forEach((loader) => loader.dispose());
        };
    }, []);

    /** Landing copy and example tiles blur out and the bar's position is recorded for the glide. Await before navigating. */
    const exitLanding = () =>
        contextSafe(() =>
            new Promise<void>((resolve) => {
                const q = gsap.utils.selector(scopeRoot());
                const text = [...q(SEL.landingText), ...q(SEL.examples)];
                const done = () => {
                    flipState.current = Flip.getState(q(SEL.smilesBar));
                    resolve();
                };
                if (!text.length) return done();
                // overwrite: an Analyze during the entrance must not be undone by the still-running entrance tweens
                if (prefersReducedMotion()) return void gsap.to(text, { autoAlpha: 0, duration: 0.2, overwrite: true, onComplete: done });
                gsap.to(text, { y: -12, autoAlpha: 0, filter: "blur(4px)", duration: 0.45, stagger: 0.05, ease: "power2.in", overwrite: true, onComplete: done });
            }),
        )();

    /** Results heading and cards leave bottom-first, then the bar's position is recorded. Await before navigating. */
    const exitResults = () =>
        contextSafe(() =>
            new Promise<void>((resolve) => {
                const q = gsap.utils.selector(scopeRoot());
                const targets = [...q(SEL.resultsText), ...q(SEL.cards)];
                const done = () => {
                    flipState.current = Flip.getState(q(SEL.smilesBar));
                    resolve();
                };
                if (!targets.length) return done();
                if (prefersReducedMotion()) return void gsap.to(targets, { autoAlpha: 0, duration: 0.2, onComplete: done });
                gsap.to(targets, { y: 16, autoAlpha: 0, duration: 0.4, stagger: { each: 0.05, from: "end" }, ease: "power2.in", onComplete: done });
            }),
        )();

    return { exitLanding, exitResults };
}
