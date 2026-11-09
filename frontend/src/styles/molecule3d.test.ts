import { afterEach, describe, expect, it } from "vitest";
import gsap from "gsap";
import { MOLECULES, mountMolecule, type MoleculeHandle } from "./molecule3d";

const handles: MoleculeHandle[] = [];
const mount = (name: string, animate: boolean) => {
    const container = document.createElement("div");
    document.body.append(container);
    const handle = mountMolecule(container, MOLECULES[name], { animate });
    handles.push(handle);
    return { container, handle };
};

afterEach(() => {
    handles.splice(0).forEach(h => h.dispose());
    document.body.replaceChildren();
});

describe.each(Object.keys(MOLECULES))("%s", (name) => {
    const molecule = MOLECULES[name];

    it("draws one sphere per atom and one bond group per bond, double bonds as two lines", () => {
        const { container } = mount(name, false);
        const scene = container.querySelector(".molecule3d__scene")!;
        expect(scene.querySelectorAll(":scope > circle")).toHaveLength(molecule.atoms.length);
        const bonds = [...scene.querySelectorAll(".molecule3d__bond")];
        // the scene is depth-sorted, so compare the line counts as multisets
        const lines = bonds.map(b => b.querySelectorAll("line").length).sort();
        expect(lines).toEqual(molecule.bonds.map(([, , order]) => order).sort());
    });

    it("projects every atom inside the viewBox", () => {
        const { container } = mount(name, false);
        for (const atom of container.querySelectorAll(".molecule3d__scene > circle")) {
            const [cx, cy] = [Number(atom.getAttribute("cx")), Number(atom.getAttribute("cy"))];
            expect(Math.abs(cx)).toBeLessThan(1);
            expect(Math.abs(cy)).toBeLessThan(1);
        }
    });

    it("shows the attribution highlight when not animated", () => {
        const { container } = mount(name, false);
        const xai = container.querySelector(".molecule3d__xai")!;
        expect(xai.getAttribute("opacity")).toBe("1.000");
        expect(xai.querySelector(".molecule3d__halo")!.getAttribute("data-tone")).toBe(molecule.groups[0].tone);
    });
});

describe("mountMolecule", () => {
    it("gives each mount its own gradient ids", () => {
        const a = mount("benzene", false).container.querySelector("radialGradient")!.id;
        const b = mount("benzene", false).container.querySelector("radialGradient")!.id;
        expect(a).not.toBe(b);
    });

    it("removes its svg on dispose", () => {
        const { container, handle } = mount("ethanol", false);
        handle.dispose();
        expect(container.querySelector("svg")).toBeNull();
    });

    it("rotates on each ticker frame while animated and stops after dispose", () => {
        const { container, handle } = mount("caffeine", true);
        const atom = container.querySelector(".molecule3d__scene > circle")!;
        const read = () => atom.getAttribute("cx");

        const before = read();
        gsap.ticker.tick();
        gsap.ticker.tick();
        const after = read();
        expect(after).not.toBe(before);

        handle.dispose();
        gsap.ticker.tick();
        expect(read()).toBe(after);
    });
});
