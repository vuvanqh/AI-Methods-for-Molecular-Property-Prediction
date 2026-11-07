import { describe, expect, it } from "vitest";
import { analysisQuery, fetchAnalysis } from "./analysis";
import { mockFetch, requestBody } from "../test/helpers";
import { aspirinResponse } from "../test/fixtures/aspirin";

describe("fetchAnalysis", () => {
    it("POSTs the SMILES as JSON with the caller's abort signal", async () => {
        const fetch = mockFetch(aspirinResponse);
        const signal = new AbortController().signal;

        await fetchAnalysis("CCO", signal);

        const [url, init] = fetch.mock.calls[0];
        expect(url).toBe("/api/predict-properties");
        expect(init).toMatchObject({ method: "POST", headers: { "Content-Type": "application/json" }, signal });
        expect(requestBody(fetch)).toEqual({ smiles: "CCO" });
    });

    it("returns the parsed response on 200", async () => {
        mockFetch(aspirinResponse);
        await expect(fetchAnalysis("CCO")).resolves.toEqual(aspirinResponse);
    });

    it.each([
        [422, "The molecule could not be processed."],
        [503, "GNN service unavailable."],
    ])("throws the backend's message on %i", async (status, message) => {
        mockFetch({ message }, status);
        await expect(fetchAnalysis("CCO")).rejects.toThrow(message);
    });

    it("falls back to the status when the error body is not JSON", async () => {
        mockFetch("Internal Server Error", 500);
        await expect(fetchAnalysis("CCO")).rejects.toThrow("Analysis failed (500).");
    });

    it("falls back to the status when the error JSON has no message (FastAPI {detail})", async () => {
        mockFetch({ detail: [{ msg: "field required" }] }, 422);
        await expect(fetchAnalysis("CCO")).rejects.toThrow("Analysis failed (422).");
    });
});

describe("analysisQuery", () => {
    it("keys the cache by SMILES", () => {
        expect(analysisQuery("CCO").queryKey).toEqual(["analysis", "CCO"]);
        expect(analysisQuery("CCO").queryKey).not.toEqual(analysisQuery("CCC").queryKey);
    });

    it("fetches that SMILES and passes TanStack's abort signal on to fetch", async () => {
        const fetch = mockFetch(aspirinResponse);
        const signal = new AbortController().signal;
        const { queryFn } = analysisQuery("CC(=O)O");

        // queryFn is normally called by TanStack; only `signal` matters to ours
        await expect((queryFn as (ctx: { signal: AbortSignal }) => Promise<unknown>)({ signal })).resolves.toEqual(aspirinResponse);
        expect(requestBody(fetch)).toEqual({ smiles: "CC(=O)O" });
        expect(fetch.mock.calls[0][1]?.signal).toBe(signal);
    });
});
