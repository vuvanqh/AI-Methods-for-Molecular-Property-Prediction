import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MoleculeOverviewCard } from "./MoleculeOverviewCard";
import { aspirinResponse } from "../../test/fixtures/aspirin";

describe("MoleculeOverviewCard", () => {
    const renderCard = () => render(
        <MoleculeOverviewCard molecule={aspirinResponse.molecule}><div data-testid="viewer"/></MoleculeOverviewCard>);

    it("renders the formula with subscripted counts", () => {
        renderCard();
        const name = screen.getByRole("heading", { level: 2 });
        expect(name).toHaveTextContent("C9H8O4");
        expect([...name.querySelectorAll("sub")].map(s => s.textContent)).toEqual(["9", "8", "4"]);
    });

    it("lists the descriptors", () => {
        renderCard();
        expect(screen.getByText(aspirinResponse.molecule.smiles)).toBeInTheDocument();
        expect(screen.getByText("Molar mass").nextElementSibling).toHaveTextContent("180.16 g/mol");
        expect(screen.getByText("Atoms (with H)").nextElementSibling).toHaveTextContent("21");
        expect(screen.getByText("Bonds (with H)").nextElementSibling).toHaveTextContent("21");
    });

    it("renders the viewer it is given and a legend of the elements present", () => {
        renderCard();
        expect(screen.getByTestId("viewer")).toBeInTheDocument();
        // aspirin has no nitrogen: the legend follows the formula instead of a fixed list
        expect(screen.getAllByRole("listitem").map(li => li.textContent)).toEqual(["C", "H", "O"]);
    });

    it("includes a salt's counter-ion and gives it a colour of its own", () => {
        const sodiumAcetate = { ...aspirinResponse.molecule, formula: "C2H3NaO2" };
        render(<MoleculeOverviewCard molecule={sodiumAcetate}><div/></MoleculeOverviewCard>);

        const items = screen.getAllByRole("listitem");
        expect(items.map(li => li.textContent)).toEqual(["C", "H", "Na", "O"]);
        expect(items[2].style.getPropertyValue("--atom-dot")).toBe("#ab5cf2");
    });
});
