import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { SimilarMoleculesCard } from "./SimilarMoleculesCard";
import { aspirinResponse } from "../../test/fixtures/aspirin";

describe("SimilarMoleculesCard", () => {
    it("renders nothing without similar molecules", () => {
        const { container } = render(<SimilarMoleculesCard similarMolecules={[]}/>);
        expect(container).toBeEmptyDOMElement();
    });

    it("lists each molecule with its score, plus the footnote", () => {
        render(<SimilarMoleculesCard similarMolecules={aspirinResponse.similar_compounds}/>);
        const rows = screen.getAllByRole("listitem");
        expect(rows.map(r => [r.querySelector(".similar-molecule-name")?.textContent, r.querySelector(".similar-molecule-score")?.textContent]))
            .toEqual([["C7H6O3", "0.88"], ["C8H9NO2", "0.43"]]);
        expect(screen.getByText("0.88")).toHaveAttribute("data-value", "0.88");
        expect(screen.getByText("Higher scores indicate closer structural relatedness.")).toBeInTheDocument();
    });
});
