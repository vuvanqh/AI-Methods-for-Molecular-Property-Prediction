import { describe, expect, it } from "vitest";
import { atomColor } from "./atomColors";

const tokens = (values: Record<string, string>) => ({ getPropertyValue: (name: string) => values[name] ?? "" });

describe("atomColor", () => {
    it("prefers the theme's token for an element the palette defines", () => {
        expect(atomColor("C", tokens({ "--atom-c": " #c2c8da " }))).toBe("#c2c8da");
    });

    it("falls back to the conventional colour for elements the theme does not define", () => {
        expect(atomColor("Na", tokens({}))).toBe("#ab5cf2");
        expect(atomColor("Cl", tokens({}))).toBe("#1ff01f");
    });

    it("still returns a colour for an unknown symbol", () => {
        expect(atomColor("Xx", tokens({}))).toMatch(/^#[0-9a-f]{6}$/);
    });
});
