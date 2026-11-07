import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PropertyItem } from "./PropertyItem";
import { PropertyResultsContainer } from "./PropertyResultsContainer";
import { PropertyPredictionCard } from "./PropertyPredictionCard";
import { toPropertyPredictions } from "../../api/mappers";
import { aspirinResponse } from "../../test/fixtures/aspirin";

const families = toPropertyPredictions(aspirinResponse.properties_xai_data);
const cyp = families[0].prediciton_results;

describe("PropertyItem", () => {
    it("shows the name, the predicted class and the confidence", () => {
        render(<PropertyItem property={cyp[0]}/>);
        const button = screen.getByRole("button");
        expect(button).toHaveTextContent("CYP1A2_Veith");
        expect(button).toHaveTextContent("True");
        expect(button).toHaveTextContent("Confidence: 0.88");
        expect(button).toHaveAttribute("aria-pressed", "false");
    });

    it("shows False for a negative prediction", () => {
        render(<PropertyItem property={cyp[1]}/>);
        expect(screen.getByText("False")).toHaveAttribute("data-value", "false");
    });

    it("carries the data-* hooks the motion module counts up from", () => {
        render(<PropertyItem property={cyp[0]}/>);
        expect(screen.getByText(/Confidence/)).toHaveAttribute("data-animate", "count");
        expect(screen.getByText(/Confidence/)).toHaveAttribute("data-value", "0.88");
    });

    it("is pressed when selected and reports its name on click", async () => {
        const onSelect = vi.fn();
        render(<PropertyItem property={cyp[0]} selected onPropertySelect={onSelect}/>);
        expect(screen.getByRole("button")).toHaveAttribute("aria-pressed", "true");
        await userEvent.click(screen.getByRole("button"));
        expect(onSelect).toHaveBeenCalledExactlyOnceWith("CYP1A2_Veith");
    });

    it("tolerates a click without a handler", async () => {
        render(<PropertyItem property={cyp[0]}/>);
        await userEvent.click(screen.getByRole("button"));
    });
});

describe("PropertyResultsContainer", () => {
    it("renders one item per result and presses only the selected one", () => {
        render(<PropertyResultsContainer selectedPropertyFamily={cyp} selectedProperty="CYP2C9_Veith"/>);
        const buttons = screen.getAllByRole("button");
        expect(buttons).toHaveLength(2);
        expect(buttons.map(b => b.getAttribute("aria-pressed"))).toEqual(["false", "true"]);
    });
});

describe("PropertyPredictionCard", () => {
    it("shows an empty state without predictions", () => {
        render(<PropertyPredictionCard propertyResult={[]}/>);
        expect(screen.getByText("No predictions available")).toBeInTheDocument();
    });

    it("starts on the first family and offers every family", () => {
        render(<PropertyPredictionCard propertyResult={families}/>);
        const select = screen.getByRole("combobox");
        expect(select).toHaveValue("CYP450 inhibition");
        expect(within(select).getAllByRole("option").map(o => o.textContent)).toEqual(families.map(f => f.property_family));
        expect(screen.getAllByRole("button").map(b => b.querySelector(".property-name")?.textContent)).toEqual(["CYP1A2_Veith", "CYP2C9_Veith"]);
    });

    it("swaps the list when another family is chosen", async () => {
        render(<PropertyPredictionCard propertyResult={families}/>);
        await userEvent.selectOptions(screen.getByRole("combobox"), "Toxicity (Tox21)");
        expect(screen.getByRole("combobox")).toHaveValue("Toxicity (Tox21)");
        expect(screen.getAllByRole("button").map(b => b.querySelector(".property-name")?.textContent)).toEqual(["NR-AR", "SR-ARE"]);
    });

    it("passes the selection through to the items", async () => {
        const onSelect = vi.fn();
        render(<PropertyPredictionCard propertyResult={families} selectedProperty="CYP1A2_Veith" onPropertySelect={onSelect}/>);
        expect(screen.getByRole("button", { name: /CYP1A2_Veith/ })).toHaveAttribute("aria-pressed", "true");
        await userEvent.click(screen.getByRole("button", { name: /CYP2C9_Veith/ }));
        expect(onSelect).toHaveBeenCalledWith("CYP2C9_Veith");
    });
});
