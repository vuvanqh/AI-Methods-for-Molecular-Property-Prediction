// Stand-in for the 3dmol package: records calls and fakes the few viewer methods MoleculeViewer uses
import { vi } from "vitest";

export type FakeAtom = { index: number, x: number, y: number, z: number };

export function makeViewer(container: HTMLElement, atomCount = 21) {
    const atoms: FakeAtom[] = Array.from({ length: atomCount }, (_, i) => ({ index: i, x: i, y: i % 3, z: 0 }));
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

export const createViewer = vi.fn((container: HTMLElement) => {
    const viewer = makeViewer(container);
    viewers.push(viewer);
    return viewer;
});
