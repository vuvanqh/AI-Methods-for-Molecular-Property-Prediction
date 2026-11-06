import { useEffect, useRef, useState } from "react";
import { createViewer, type GLViewer } from "3dmol";

const THEMED_ELEMENTS = ["C", "O", "N", "H"];
const SVG_NS = "http://www.w3.org/2000/svg";

type MoleculeViewerProps = {
    molblock: string,
    property?: string | null,
    scores?: Record<string, number | null> // XAI atom attributions for `property`, indexed like the molblock atoms
}

// Default export so the page can React.lazy() it: 3Dmol is large and only needed on results
export default function MoleculeViewer({ molblock, property, scores }: MoleculeViewerProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const overlayRef = useRef<SVGSVGElement>(null);
    const viewerRef = useRef<GLViewer | null>(null);
    const [spinning, setSpinning] = useState(() => !window.matchMedia("(prefers-reduced-motion: reduce)").matches);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;
        const viewer = createViewer(container, { backgroundColor: "black", backgroundAlpha: 0, antialias: true });
        viewer.addModel(molblock, "sdf");
     
        const css = getComputedStyle(container);
        const map = Object.fromEntries(THEMED_ELEMENTS.map(el => [el, css.getPropertyValue(`--atom-${el.toLowerCase()}`).trim()]));
        const colorscheme = { prop: "elem", map };
        viewer.setStyle({}, { stick: { radius: 0.14, colorscheme }, sphere: { scale: 0.24, colorscheme } });
        viewer.zoomTo();
        viewer.render();
        viewerRef.current = viewer;

        return () => {
            viewer.spin(false);
            viewer.clear();
            viewerRef.current = null;
            container.replaceChildren(); // drop the WebGL canvas so remounts don't stack canvases
        };
    }, [molblock]);

    useEffect(() => {
        viewerRef.current?.spin(spinning ? "y" : false, 0.4);
    }, [molblock, spinning]);

    // XAI highlight: a ring + score per contributing atom, drawn in an SVG overlay that follows the
    // atoms' screen positions every frame (rotation, drag, zoom), so the atom's own colour stays visible
    useEffect(() => {
        const viewer = viewerRef.current;
        const overlay = overlayRef.current;
        if (!viewer || !overlay || !scores) return;

        const values = Object.values(scores).filter((v): v is number => v !== null);
        const max = Math.max(...values.map(Math.abs)) || 1;
       
        const labelled = new Set(values.map(Math.abs).sort((a, b) => b - a).slice(0, 4));
        const marks = viewer.getModel().selectedAtoms({}).flatMap(atom => {
            const s = scores[atom.index ?? -1]; // hydrogens have no score (null): no mark
            if (s == null) return [];

            // sqrt lifts mid-strength atoms so more than the single strongest one stands out; order is kept
            const t = Math.sqrt(Math.abs(s) / max);
            const g = document.createElementNS(SVG_NS, "g");

            g.style.opacity = (0.15 + 0.85 * t).toFixed(2);
            const ring = document.createElementNS(SVG_NS, "circle");
            ring.setAttribute("stroke-width", (0.75 + 1.75 * t).toFixed(2));
            const label = document.createElementNS(SVG_NS, "text");
            if (labelled.has(Math.abs(s))) label.textContent = `${s > 0 ? "+" : "−"}${Math.abs(s).toExponential(1)}`;
            g.append(ring, label);
            overlay.append(g);
            return [{ atom, t, g, ring, label }];
        });

        let frame = 0;
        const draw = () => {
            const center = { x: 0, y: 0, z: 0 };
            marks.forEach(({ atom }) => { center.x += atom.x ?? 0; center.y += atom.y ?? 0; center.z += atom.z ?? 0; });
            if (marks.length) { center.x /= marks.length; center.y /= marks.length; center.z /= marks.length; }


            const [c, ...axes] = viewer.modelToScreen([center,
                { ...center, x: center.x + 1 }, { ...center, y: center.y + 1 }, { ...center, z: center.z + 1 }]);
            const scale = Math.max(...axes.map(p => Math.hypot(p.x - c.x, p.y - c.y)));
            const points = viewer.modelToScreen(marks.map(({ atom }) => ({ x: atom.x, y: atom.y, z: atom.z })));

            // modelToScreen returns document coordinates (canvas rect + page scroll); the overlay needs canvas-relative ones
            const box = overlay.getBoundingClientRect();
            const [dx, dy] = [box.left + window.scrollX, box.top + window.scrollY];
            marks.forEach(({ t, g, ring, label }, i) => {

                const r = scale * (0.5 + 0.12 * t); 
                g.setAttribute("transform", `translate(${(points[i].x - dx).toFixed(1)} ${(points[i].y - dy).toFixed(1)})`);
                ring.setAttribute("r", r.toFixed(1));
                label.setAttribute("x", (r * 0.75 + 3).toFixed(1));
                label.setAttribute("y", (-r * 0.75 - 2).toFixed(1));
            });
            frame = requestAnimationFrame(draw);
        };
        draw();

        return () => {
            cancelAnimationFrame(frame);
            overlay.replaceChildren();
        };
    }, [molblock, scores]);

    return <>
        <div ref={containerRef} className="molecule-overview__canvas" role="img"
            aria-label={scores ? `3D model of the molecule, atoms marked by their attribution to ${property}` : "Interactive 3D model of the molecule"}/>
        <svg ref={overlayRef} className="molecule-overview__overlay" aria-hidden="true"/>
        <div className="molecule-overview__toolbar">
            <button type="button" className="molecule-overview__tool" aria-pressed={spinning}
                aria-label={spinning ? "Pause rotation" : "Resume rotation"} title={spinning ? "Pause rotation" : "Resume rotation"}
                onClick={() => setSpinning(s => !s)}>
                {spinning ? "❚❚" : "▶"}
            </button>
        </div>
        {scores && <div className="molecule-overview__attribution">
            <span>Atom attribution · <strong>{property}</strong></span>
            <span className="molecule-overview__attribution-scale" aria-hidden="true">weak <i/> strong impact</span>
        </div>}
    </>;
}
