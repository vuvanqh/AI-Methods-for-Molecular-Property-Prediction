import { expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "./App";

it("mounts the router on the landing page", async () => {
    render(<App/>);
    expect(await screen.findByRole("button", { name: "Analyze Molecule" })).toBeInTheDocument();
});
