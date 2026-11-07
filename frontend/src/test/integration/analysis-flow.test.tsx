// The whole app on its real routes: page, query cache, mappers and cards together. Only fetch and 3Dmol are fake.
import { describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockFetch, mockPendingFetch, renderApp, requestBody } from "../helpers";
import { aspirinResponse } from "../fixtures/aspirin";

const HERO = /Understand molecules/;
const input = () => screen.getByRole("textbox");
const analyzeButton = () => screen.getByRole("button", { name: "Analyze Molecule" });
const results = () => screen.findByRole("heading", { name: "Scientific Interpretation" });

describe("analysis flow", () => {
    it("starts on the landing page without results", async () => {
        mockFetch(aspirinResponse);
        renderApp("/");
        expect(await screen.findByRole("heading", { level: 1, name: HERO })).toBeInTheDocument();
        expect(analyzeButton()).toBeEnabled();
        expect(screen.queryByText("Loading molecular structure…")).not.toBeInTheDocument();
    });

    it("submits a SMILES, shows the loading state, then every card", async () => {
        const fetch = mockPendingFetch();
        const { router } = renderApp("/");

        await userEvent.type(input(), "  CCO  {Enter}");

        await waitFor(() => expect(router.state.location.pathname).toBe("/CCO"));
        // the query starts when the results page mounts, which is when the loading state appears
        expect(await screen.findByText("Loading molecular structure…")).toHaveAttribute("aria-busy", "true");
        expect(fetch).toHaveBeenCalledOnce();
        expect(requestBody(fetch)).toEqual({ smiles: "CCO" });
        expect(screen.queryByRole("heading", { level: 1 })).not.toBeInTheDocument(); // landing hero gone
        expect(analyzeButton()).toBeDisabled();
    });

    it("renders the mapped response", async () => {
        mockFetch(aspirinResponse);
        renderApp("/CCO");
        await results();

        expect(screen.getByText("Molar mass").nextElementSibling).toHaveTextContent("180.16 g/mol");
        expect(screen.getByRole("combobox")).toHaveValue("CYP450 inhibition");
        expect(screen.getByRole("button", { name: /CYP1A2_Veith/ })).toHaveTextContent("Confidence: 0.88");
        expect(screen.getByText("Similar Molecules & Evidence")).toBeInTheDocument();
        expect(screen.getByText(aspirinResponse.llm_summary)).toBeInTheDocument();
        expect(screen.getAllByRole("link").map(l => l.textContent)).toEqual(["1", "2"]); // 3 sources, deduplicated to 2
        expect(await screen.findByRole("img", { name: "Interactive 3D model of the molecule" })).toBeInTheDocument();
    });

    it("decodes a deep link before sending it", async () => {
        const fetch = mockFetch(aspirinResponse);
        renderApp(`/${encodeURIComponent("CC(=O)O")}`);
        await results();
        expect(requestBody(fetch)).toEqual({ smiles: "CC(=O)O" });
    });

    it("toggles a property's atom attribution in the viewer, and keeps it across families", async () => {
        mockFetch(aspirinResponse);
        renderApp("/CCO");
        await results();
        await screen.findByRole("img", { name: "Interactive 3D model of the molecule" });

        await userEvent.click(screen.getByRole("button", { name: /CYP1A2_Veith/ }));
        expect(screen.getByRole("button", { name: /CYP1A2_Veith/ })).toHaveAttribute("aria-pressed", "true");
        expect(screen.getByRole("img", { name: /attribution to CYP1A2_Veith/ })).toBeInTheDocument();
        expect(screen.getByText(/Atom attribution/)).toHaveTextContent("Atom attribution · CYP1A2_Veith");

        await userEvent.selectOptions(screen.getByRole("combobox"), "Toxicity (Tox21)");
        expect(screen.getByText(/Atom attribution/)).toHaveTextContent("CYP1A2_Veith");
        await userEvent.selectOptions(screen.getByRole("combobox"), "CYP450 inhibition");

        await userEvent.click(screen.getByRole("button", { name: /CYP1A2_Veith/ }));
        expect(screen.queryByText(/Atom attribution/)).not.toBeInTheDocument();
    });

    it("New Analysis returns to an empty landing page", async () => {
        mockFetch(aspirinResponse);
        const { router } = renderApp("/");
        await userEvent.type(input(), "CCO{Enter}");
        await results();

        await userEvent.click(screen.getByRole("button", { name: "+ New Analysis" }));

        await waitFor(() => expect(router.state.location.pathname).toBe("/"));
        expect(input()).toHaveValue("");
        expect(await screen.findByRole("heading", { level: 1, name: HERO })).toBeInTheDocument();
        expect(analyzeButton()).toBeEnabled();
    });

    it("analyzing the same molecule again is served from the cache", async () => {
        const fetch = mockFetch(aspirinResponse);
        renderApp("/");
        await userEvent.type(input(), "CCO{Enter}");
        await results();
        await userEvent.click(screen.getByRole("button", { name: "+ New Analysis" }));
        await screen.findByRole("heading", { level: 1, name: HERO });

        await userEvent.type(input(), "CCO{Enter}");
        await results();
        expect(fetch).toHaveBeenCalledOnce();
    });

    it("New Analysis while loading aborts the pending request", async () => {
        const fetch = mockPendingFetch();
        const { router } = renderApp("/");
        await userEvent.type(input(), "CCO{Enter}");
        await screen.findByText("Loading molecular structure…");

        await userEvent.click(screen.getByRole("button", { name: "+ New Analysis" }));

        await waitFor(() => expect(router.state.location.pathname).toBe("/"));
        // the router updates first; the abort follows once React has unmounted the results page
        await waitFor(() => expect((fetch.mock.calls[0][1] as RequestInit).signal!.aborted).toBe(true));
    });

    it("New Analysis does nothing on the landing page", async () => {
        mockFetch(aspirinResponse);
        const { router } = renderApp("/");
        await userEvent.type(input(), "CCO");
        await userEvent.click(screen.getByRole("button", { name: "+ New Analysis" }));
        expect(router.state.location.pathname).toBe("/");
        expect(input()).toHaveValue("CCO");
    });

    it("leaves out the similar-molecules and interpretation cards when the response has none", async () => {
        mockFetch({ ...aspirinResponse, similar_compounds: [], llm_summary: "" });
        renderApp("/CCO");
        await screen.findByRole("combobox");
        expect(screen.queryByText("Similar Molecules & Evidence")).not.toBeInTheDocument();
        expect(screen.queryByRole("heading", { name: "Scientific Interpretation" })).not.toBeInTheDocument();
    });
});
