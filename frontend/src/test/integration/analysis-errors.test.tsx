import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import { mockFetch, renderApp } from "../helpers";

describe("analysis errors", () => {
    it.each([
        [422, { message: "The molecule could not be processed." }, "The molecule could not be processed."],
        [503, { message: "GNN service unavailable." }, "GNN service unavailable."],
        [500, "oops", "Analysis failed (500)."],
    ])("shows the %i error in an alert", async (status, body, message) => {
        const fetch = mockFetch(body, status);
        renderApp("/C1CC");
        expect(await screen.findByRole("alert")).toHaveTextContent(message);
        expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
        expect(fetch).toHaveBeenCalledOnce(); // retry is off: the error shows at once
    });

    it("shows a network failure in an alert", async () => {
        vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
        renderApp("/CCO");
        expect(await screen.findByRole("alert")).toHaveTextContent("Failed to fetch");
    });
});
