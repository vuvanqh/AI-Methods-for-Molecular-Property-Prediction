import { expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Navbar } from "./Navbar";

it("shows the title and calls onNewAnalysis on click", async () => {
    const onNewAnalysis = vi.fn();
    render(<Navbar onNewAnalysis={onNewAnalysis}/>);
    expect(screen.getByText("Molecular Property Analysis")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "+ New Analysis" }));
    expect(onNewAnalysis).toHaveBeenCalledOnce();
});
