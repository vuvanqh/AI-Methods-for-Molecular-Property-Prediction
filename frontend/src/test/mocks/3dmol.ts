// Stand-in for the 3dmol package: records calls and fakes the few viewer methods MoleculeViewer uses
import { vi } from "vitest";

export type FakeAtom = { index: number, x: number, y: number, z: number, elem: string };

// includes an element outside the themed palette, so the colour fallback is exercised
const ELEMENTS = ["C", "O", "N", "S"];

export function makeViewer(container: HTMLElement, atomCount = 21) {
    const atoms: FakeAtom[] = Array.from({ length: atomCount }, (_, i) => ({ index: i, x: i, y: i % 3, z: 0, elem: ELEMENTS[i % ELEMENTS.length] }));
    container.append(document.createElement("canvas"));
    return {
        container,
        atoms,
        addModel: vi.fn(),
        setStyle: vi.fn(),
        zoomTo: vi.fn(),
        render: vi.fn(),
        spin: vi.fn(),
        clear: vi.fn(),
        getModel: () => ({ selectedAtoms: () => atoms }),
        modelToScreen: vi.fn((points: { x?: number, y?: number }[]) => points.map(p => ({ x: (p.x ?? 0) * 10, y: (p.y ?? 0) * 10 }))),
    };
}

export type FakeViewer = ReturnType<typeof makeViewer>;

export const viewers: FakeViewer[] = [];

// the handful of Jmol colours the tests assert on; the real table covers the periodic table
export const elementColors = {
    defaultColor: 0xff1493,
    Jmol: { H: 0xffffff, C: 0x909090, N: 0x3050f8, O: 0xff0d0d, S: 0xffff30, Na: 0xab5cf2, Cl: 0x1ff01f } as Record<string, number>,
};

export const createViewer = vi.fn((container: HTMLElement) => {
    const viewer = makeViewer(container);
    viewers.push(viewer);
    return viewer;
});
