import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { SmilesInput } from "./SmilesInput";
import { renderWithRouter } from "../../test/helpers";

// SmilesInput is controlled; this host keeps the value in state like AnalysisPage does
function Host({ onAnalyze, disabled = false, initial = "" }: { onAnalyze: (s: string) => void, disabled?: boolean, initial?: string }) {
    const [value, setValue] = useState(initial);
    return <SmilesInput inputSmiles={value} onChange={setValue} onAnalyze={onAnalyze} disabled={disabled}/>;
}

describe("SmilesInput", () => {
    it("labels the input", () => {
        renderWithRouter(<Host onAnalyze={vi.fn()}/>);
        expect(screen.getByLabelText("Enter a molecular structure in SMILES format")).toHaveAttribute("id", "smiles-input");
    });

    it("reports every keystroke through onChange", async () => {
        const onChange = vi.fn();
        renderWithRouter(<SmilesInput inputSmiles="" onChange={onChange} onAnalyze={vi.fn()} disabled={false}/>);
        await userEvent.type(screen.getByRole("textbox"), "CO");
        expect(onChange.mock.calls).toEqual([["C"], ["O"]]);
    });

    it("submits the trimmed SMILES from the button", async () => {
        const onAnalyze = vi.fn();
        renderWithRouter(<Host onAnalyze={onAnalyze}/>);
        await userEvent.type(screen.getByRole("textbox"), "  CCO  ");
        await userEvent.click(screen.getByRole("button", { name: "Analyze Molecule" }));
        expect(onAnalyze).toHaveBeenCalledExactlyOnceWith("CCO");
    });

    it("submits on Enter", async () => {
        const onAnalyze = vi.fn();
        renderWithRouter(<Host onAnalyze={onAnalyze}/>);
        await userEvent.type(screen.getByRole("textbox"), "c1ccccc1{Enter}");
        expect(onAnalyze).toHaveBeenCalledExactlyOnceWith("c1ccccc1");
    });

    it.each(["", "   "])("ignores a blank submit (%j)", async (input) => {
        const onAnalyze = vi.fn();
        renderWithRouter(<Host onAnalyze={onAnalyze} initial={input}/>);
        await userEvent.click(screen.getByRole("button", { name: "Analyze Molecule" }));
        expect(onAnalyze).not.toHaveBeenCalled();
    });

    it("disables the button when disabled", () => {
        renderWithRouter(<Host onAnalyze={vi.fn()} disabled/>);
        expect(screen.getByRole("button", { name: "Analyze Molecule" })).toBeDisabled();
    });
});
