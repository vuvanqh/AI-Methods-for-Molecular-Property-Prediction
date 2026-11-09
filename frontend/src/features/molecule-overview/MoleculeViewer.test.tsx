import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import MoleculeViewer from "./MoleculeViewer";
import { createViewer, viewers } from "../../test/mocks/3dmol";
import { setReducedMotion } from "../../test/media";
import { aspirinResponse, ASPIRIN_MOLBLOCK } from "../../test/fixtures/aspirin";

const viewer = () => viewers[viewers.length - 1];
const overlay = (container: HTMLElement) => container.querySelector("svg.molecule-overview__overlay")!;

beforeEach(() => {
    viewers.length = 0;
    createViewer.mockClear();
});

describe("MoleculeViewer", () => {
    it("loads the molblock and renders it once", () => {
        render(<MoleculeViewer molblock={ASPIRIN_MOLBLOCK}/>);
        expect(createViewer).toHaveBeenCalledOnce();
        expect(viewer().addModel).toHaveBeenCalledWith(ASPIRIN_MOLBLOCK, "sdf");
        expect(viewer().setStyle).toHaveBeenCalledOnce();
        expect(viewer().zoomTo).toHaveBeenCalledOnce();
        expect(viewer().render).toHaveBeenCalledOnce();
    });

    it("spins by default and the toggle pauses and resumes it", async () => {
        setReducedMotion(false);
        render(<MoleculeViewer molblock={ASPIRIN_MOLBLOCK}/>);
        expect(viewer().spin).toHaveBeenLastCalledWith("y", 0.4);

        await userEvent.click(screen.getByRole("button", { name: "Pause rotation" }));
        expect(viewer().spin).toHaveBeenLastCalledWith(false, 0.4);
        const resume = screen.getByRole("button", { name: "Resume rotation" });
        expect(resume).toHaveAttribute("aria-pressed", "false");

        await userEvent.click(resume);
        expect(viewer().spin).toHaveBeenLastCalledWith("y", 0.4);
    });

    it("starts paused when the user prefers reduced motion", () => {
        render(<MoleculeViewer molblock={ASPIRIN_MOLBLOCK}/>);
        expect(viewer().spin).toHaveBeenLastCalledWith(false, 0.4);
        expect(screen.getByRole("button", { name: "Resume rotation" })).toBeInTheDocument();
    });

    it("shows no attribution without scores", () => {
        const { container } = render(<MoleculeViewer molblock={ASPIRIN_MOLBLOCK}/>);
        expect(screen.getByRole("img")).toHaveAccessibleName("Interactive 3D model of the molecule");
        expect(screen.queryByText(/Atom attribution/)).not.toBeInTheDocument();
        expect(overlay(container).childElementCount).toBe(0);
    });

    it("marks every scored atom and labels only the four strongest", () => {
        const scores = aspirinResponse.properties_xai_data.atomscores["NR-AR"];
        const { container } = render(<MoleculeViewer molblock={ASPIRIN_MOLBLOCK} property="NR-AR" scores={scores}/>);

        expect(screen.getByRole("img")).toHaveAccessibleName("3D model of the molecule, atoms marked by their attribution to NR-AR");
        expect(screen.getByText("NR-AR").closest(".molecule-overview__attribution")).toBeInTheDocument();

        const marks = overlay(container).querySelectorAll("g");
        expect(marks).toHaveLength(13); // 13 heavy atoms; the 8 hydrogens are null
        const labels = [...overlay(container).querySelectorAll("text")].map(t => t.textContent).filter(Boolean);
        expect(labels).toHaveLength(4);
        for (const label of labels) expect(label).toMatch(/^[+−]\d\.\de[-+]\d+$/);

        // positioned from modelToScreen: the fake maps atom (x, y) to (10x, 10y)
        expect(marks[1].getAttribute("transform")).toBe("translate(10.0 10.0)");
    });

    it("handles all-zero scores without NaN", () => {
        const { container } = render(<MoleculeViewer molblock={ASPIRIN_MOLBLOCK} property="x" scores={{ 0: 0, 1: 0, 2: null }}/>);
        const marks = overlay(container).querySelectorAll("g");
        expect(marks).toHaveLength(2);
        for (const g of marks) expect(g.style.opacity).toBe("0.15");
        for (const ring of overlay(container).querySelectorAll("circle")) expect(ring.getAttribute("r")).not.toBe("NaN");
    });

    it("clears the marks when the scores go away", () => {
        const scores = aspirinResponse.properties_xai_data.atomscores["NR-AR"];
        const { container, rerender } = render(<MoleculeViewer molblock={ASPIRIN_MOLBLOCK} property="NR-AR" scores={scores}/>);
        rerender(<MoleculeViewer molblock={ASPIRIN_MOLBLOCK} property={null}/>);
        expect(overlay(container).childElementCount).toBe(0);
    });

    it("tears the viewer down on unmount", () => {
        const cancel = vi.spyOn(window, "cancelAnimationFrame");
        const scores = aspirinResponse.properties_xai_data.atomscores["NR-AR"];
        const { unmount } = render(<MoleculeViewer molblock={ASPIRIN_MOLBLOCK} property="NR-AR" scores={scores}/>);
        const v = viewer();
        expect(v.container.querySelector("canvas")).not.toBeNull();

        unmount();
        expect(v.spin).toHaveBeenLastCalledWith(false);
        expect(v.clear).toHaveBeenCalledOnce();
        expect(v.container.childElementCount).toBe(0);
        expect(cancel).toHaveBeenCalled();
    });

    it("builds a new viewer for a new molblock", () => {
        const { rerender } = render(<MoleculeViewer molblock={ASPIRIN_MOLBLOCK}/>);
        const first = viewer();
        rerender(<MoleculeViewer molblock="other"/>);
        expect(first.clear).toHaveBeenCalledOnce();
        expect(createViewer).toHaveBeenCalledTimes(2);
        expect(viewer().addModel).toHaveBeenCalledWith("other", "sdf");
    });
});
