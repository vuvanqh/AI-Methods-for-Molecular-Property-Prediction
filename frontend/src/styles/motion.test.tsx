import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useAnalysisMotion } from "./motion";
import { setReducedMotion } from "../test/media";

// the markup hooks motion.ts looks for (see its header comment)
function page({ landing = true, results = false } = {}) {
    document.body.innerHTML = `
        <nav class="navbar"></nav>
        <div class="molecule-field__item" data-molecule="benzene"></div>
        <div class="molecule-field__item" data-molecule="caffeine"></div>
        <div class="molecule-field__item" data-molecule="unknown"></div>
        ${landing ? '<header class="landing-hero"><h1>Landing</h1></header>' : ""}
        ${results ? '<div class="analysis-results"><header class="analysis-hero"><h2>Results</h2></header><section class="card"></section></div>' : ""}
        <section class="smiles-input-container"></section>
        ${landing ? '<div class="smiles-examples"><h2 class="smiles-examples__label">Examples</h2><ul><li class="smiles-examples__item"><button><span class="smiles-example__name">Aspirin</span><span class="smiles-example__smiles">CC(=O)Oc1ccccc1C(=O)O</span></button></li></ul></div>' : ""}`;
}

const molecules = () => [...document.querySelectorAll(".molecule-field__item")].map(el => el.querySelector("svg.molecule3d") !== null);

afterEach(() => {
    document.body.replaceChildren();
});

describe.each([true, false])("useAnalysisMotion (reduced motion: %s)", (reduced) => {
    beforeEach(() => setReducedMotion(reduced));

    it("mounts the background molecules into known slots and disposes them on unmount", () => {
        page();
        const { unmount } = renderHook(() => useAnalysisMotion({ analysisStarted: false }));
        expect(molecules()).toEqual([true, true, false]);
        unmount();
        expect(molecules()).toEqual([false, false, false]);
    });

    it("exitLanding resolves once the landing copy has left", async () => {
        page();
        const { result } = renderHook(() => useAnalysisMotion({ analysisStarted: false }));
        await act(() => result.current.exitLanding());
        expect(document.querySelector<HTMLElement>(".landing-hero h1")!.style.opacity).toBe("0");
    });

    it("exitLanding takes the example tiles out with the landing copy", async () => {
        page();
        const { result } = renderHook(() => useAnalysisMotion({ analysisStarted: false }));
        await act(() => result.current.exitLanding());
        expect([...document.querySelectorAll<HTMLElement>(".smiles-examples__label, .smiles-examples__item")].map(el => el.style.opacity)).toEqual(["0", "0"]);
    });

    it("exitLanding resolves at once when there is no landing copy", async () => {
        page({ landing: false });
        const { result } = renderHook(() => useAnalysisMotion({ analysisStarted: false }));
        await act(() => result.current.exitLanding());
    });

    it("exitResults resolves once the results have left, then the next view takes over", async () => {
        page({ landing: false, results: true });
        const { result, rerender } = renderHook(({ started }) => useAnalysisMotion({ analysisStarted: started }), { initialProps: { started: true } });
        // only resolution is checked: exiting right after mount races the cards' entrance tween
        await act(() => result.current.exitResults());
        rerender({ started: false }); // consumes the recorded Flip state
    });

    it("mounts the analysis loader whether it is already there or appears later, and removes it on unmount", async () => {
        page({ landing: false });
        const loader = () => Object.assign(document.createElement("div"), { className: "analysis-loader" });
        const early = loader();
        early.dataset.animate = "loader";
        document.body.append(early);
        const { unmount } = renderHook(() => useAnalysisMotion({ analysisStarted: true }));
        expect(early.querySelector(".analysis-loader__stage")).not.toBeNull();

        const late = loader();
        late.dataset.animate = "loader";
        document.body.append(late);
        await waitFor(() => expect(late.querySelector(".analysis-loader__stage")).not.toBeNull());

        unmount();
        expect(document.querySelector(".analysis-loader__stage")).toBeNull();
    });

    it("exitResults resolves at once when there are no results", async () => {
        page();
        const { result } = renderHook(() => useAnalysisMotion({ analysisStarted: false }));
        await act(() => result.current.exitResults());
    });
});

describe("useAnalysisMotion count-up and re-stagger (full motion)", () => {
    beforeEach(() => setReducedMotion(false));

    const list = () => {
        const ul = document.createElement("ul");
        ul.dataset.animate = "stagger";
        ul.innerHTML = '<li><span data-animate="count" data-value="0.88">Confidence: <!---->0.88</span></li>';
        // React renders "Confidence: {n}" as two text nodes; mimic that so the number has its own node
        const span = ul.querySelector("span")!;
        span.replaceChildren(document.createTextNode("Confidence: "), document.createTextNode("0.88"));
        return { ul, number: () => span.childNodes[1].nodeValue };
    };

    it("counts numbers up from zero when a list mounts later", async () => {
        page();
        renderHook(() => useAnalysisMotion({ analysisStarted: false }));
        const { ul, number } = list();
        document.body.append(ul);

        await waitFor(() => expect(number()).not.toBe("0.88"));
        await waitFor(() => expect(number()).toBe("0.88"), { timeout: 3000 });
    });

    it("re-staggers a card's list when its family select changes", async () => {
        page();
        renderHook(() => useAnalysisMotion({ analysisStarted: false }));
        const card = document.createElement("section");
        card.className = "card";
        const select = document.createElement("select");
        select.className = "property-family-select";
        card.append(select);
        document.body.append(card);
        const { ul, number } = list();
        await waitFor(() => expect(number()).toBe("0.88")); // let the mount animation settle first
        card.append(ul);
        await waitFor(() => expect(number()).toBe("0.88"), { timeout: 3000 });

        select.dispatchEvent(new Event("change", { bubbles: true }));
        await waitFor(() => expect(number()).not.toBe("0.88"));
    });

    it("ignores change events from other elements", async () => {
        page();
        renderHook(() => useAnalysisMotion({ analysisStarted: false }));
        const input = document.createElement("input");
        document.body.append(input);
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });
});
