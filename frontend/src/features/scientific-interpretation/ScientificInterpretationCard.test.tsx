import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ScientificInterpretationCard } from "./ScientificInterpretationCard";

const SOURCES = ["https://example.org/a", "https://example.org/b"];

describe("ScientificInterpretationCard", () => {
    it("renders nothing without an interpretation", () => {
        const { container } = render(<ScientificInterpretationCard scientificInterpretation={{ interpretation: "", sources: SOURCES }}/>);
        expect(container).toBeEmptyDOMElement();
    });

    it("shows the interpretation with numbered source links that open safely in a new tab", () => {
        render(<ScientificInterpretationCard scientificInterpretation={{ interpretation: "It is aspirin.", sources: SOURCES }}/>);
        expect(screen.getByRole("heading", { name: "Scientific Interpretation" })).toBeInTheDocument();
        expect(screen.getByText("It is aspirin.")).toBeInTheDocument();
        const links = screen.getAllByRole("link");
        expect(links.map(l => [l.textContent, l.getAttribute("href")])).toEqual([["1", SOURCES[0]], ["2", SOURCES[1]]]);
        for (const link of links) {
            expect(link).toHaveAttribute("target", "_blank");
            expect(link).toHaveAttribute("rel", "noopener noreferrer");
        }
    });

    it("shows no links without sources", () => {
        render(<ScientificInterpretationCard scientificInterpretation={{ interpretation: "It is aspirin.", sources: [] }}/>);
        expect(screen.queryByRole("link")).not.toBeInTheDocument();
    });
});
