/**
 * One colour per element, for the 3D viewer and the legend dots alike.
 *
 * A `--atom-<symbol>` CSS token wins when the theme defines one (C, N, O, H today), so the
 * palette stays ours for the elements that dominate drug-like molecules. Everything else —
 * sulfur, halogens, salt counter-ions — falls back to 3Dmol's Jmol table instead of a fixed
 * list, so an element nobody anticipated is still drawn in its conventional colour.
 */
import { elementColors } from "3dmol";

type Tokens = Pick<CSSStyleDeclaration, "getPropertyValue">;

const hex = (value: number) => `#${value.toString(16).padStart(6, "0")}`;

export function atomColor(element: string, tokens?: Tokens): string {
    const css = tokens ?? getComputedStyle(document.documentElement);
    const themed = css.getPropertyValue(`--atom-${element.toLowerCase()}`).trim();
    if (themed) return themed;

    const jmol = elementColors.Jmol[element];
    return hex(typeof jmol === "number" ? jmol : elementColors.defaultColor);
}
