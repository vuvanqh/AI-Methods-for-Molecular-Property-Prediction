import { afterEach, describe, expect, it, vi } from "vitest";
import gsap from "gsap";
import { AT, mountLoader, type LoaderHandle } from "./loader";
import { MOLECULES } from "./molecule3d";

const handles: LoaderHandle[] = [];
const mount = (molecule: string, animate = true) => {
    const host = document.createElement("div");
    host.dataset.molecule = molecule;
    host.dataset.smiles = "CC(=O)OC1=CC=CC=C1C(=O)O";
    document.body.append(host);
    const handle = mountLoader(host, { animate });
    handles.push(handle);
    return { host, handle };
};

// index of the one element in the group that is shown, or -1 / several when out of sync
const shown = (els: NodeListOf<HTMLElement>) => [...els].flatMap((el, i) => (el.style.opacity === "1" && el.style.visibility !== "hidden" ? [i] : []));

afterEach(() => {
    handles.splice(0).forEach(h => h.dispose());
    document.body.replaceChildren();
    vi.restoreAllMocks();
});

describe("mountLoader", () => {
    it("draws every heavy atom and leaves hydrogens out", () => {
        const { host } = mount("aspirin");
        const heavy = MOLECULES.aspirin.atoms.filter(a => a.el !== "H").length;
        expect(host.querySelectorAll(".analysis-loader__atom")).toHaveLength(heavy);
        expect(host.querySelectorAll(".analysis-loader__atom text")).toHaveLength(4); // the 4 oxygens are labelled
    });

    it("keeps every atom inside the viewBox", () => {
        for (const name of ["aspirin", "paracetamol", "salicylic"]) {
            const { host } = mount(name);
            for (const c of host.querySelectorAll(".analysis-loader__atom circle")) {
                expect(Number(c.getAttribute("cx"))).toBeGreaterThan(0);
                expect(Number(c.getAttribute("cx"))).toBeLessThan(400);
                expect(Number(c.getAttribute("cy"))).toBeGreaterThan(0);
                expect(Number(c.getAttribute("cy"))).toBeLessThan(290);
            }
        }
    });

    it.each([
        [AT.read + 0.5, 0],
        [AT.pass + 1, 1],
        [AT.score + 0.5, 2],
        [AT.pool + 1, 3],
    ])("at %ss shows phase %i: its bar and caption together, nothing else", (time, phase) => {
        const { host, handle } = mount("aspirin");
        handle.timeline.pause(time);
        expect(shown(host.querySelectorAll(".analysis-loader__caption"))).toEqual([phase]);
        expect(shown(host.querySelectorAll(".analysis-loader__seg > span"))).toEqual([phase]);
    });

    it("keeps bar and caption in step on the second loop too", () => {
        const { host, handle } = mount("aspirin");
        handle.timeline.pause(AT.end + AT.score + 0.5); // repeat: -1, so this is the next loop's scoring phase
        expect(shown(host.querySelectorAll(".analysis-loader__caption"))).toEqual([2]);
        expect(shown(host.querySelectorAll(".analysis-loader__seg > span"))).toEqual([2]);
    });

    it("holds the attribution frame without motion", () => {
        const { host, handle } = mount("aspirin", false);
        expect(handle.timeline.paused()).toBe(true);
        expect(shown(host.querySelectorAll(".analysis-loader__caption"))).toEqual([2]);
        expect(host.querySelector(".analysis-loader__xai")!.getAttribute("r")).not.toBe("0");
    });

    it("shows the SMILES and elapsed time, and flags a slow request", () => {
        const now = vi.spyOn(performance, "now").mockReturnValue(1000);
        const { host } = mount("aspirin");
        expect(host.querySelector(".analysis-loader__clock")!.textContent).toBe("CC(=O)OC1=CC=CC=C1C(=O)O · 0.0 s");
        expect(host.dataset.slow).toBeUndefined();

        now.mockReturnValue(1000 + 13_000);
        gsap.ticker.tick();
        expect(host.querySelector(".analysis-loader__clock")!.textContent).toBe("CC(=O)OC1=CC=CC=C1C(=O)O · 13.0 s");
        expect(host.dataset.slow).toBe("");
    });

    it("cleans up by itself once its host leaves the page", () => {
        const { host, handle } = mount("aspirin");
        host.remove();
        gsap.ticker.tick();
        expect(host.querySelector(".analysis-loader__stage")).toBeNull();
        expect(gsap.globalTimeline.getChildren(false, false, true)).not.toContain(handle.timeline);
    });

    it("leaves the host's own content alone", () => {
        const host = document.createElement("div");
        host.innerHTML = '<p role="status">Loading molecular structure…</p>';
        document.body.append(host);
        handles.push(mountLoader(host, { animate: false }));
        expect(host.querySelector("p")!.textContent).toBe("Loading molecular structure…");
        expect([...host.children].filter(c => c.getAttribute("aria-hidden") !== "true")).toHaveLength(1);
    });
});
