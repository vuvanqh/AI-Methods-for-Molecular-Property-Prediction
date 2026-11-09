import { queryOptions } from "@tanstack/react-query";
import type { PredictionResponse } from "../types/molecule";

export async function fetchAnalysis(smiles: string, signal?: AbortSignal): Promise<PredictionResponse> {
    const response = await fetch("/api/predict-properties", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
        signal,
    });
    if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.message ?? `Analysis failed (${response.status}).`);
    }
    return response.json();
}

// One cache entry per SMILES. TanStack hands queryFn an AbortSignal; passing it on to fetch lets
// the query cancel the request when the last component using it unmounts mid-request.
export const analysisQuery = (smiles: string) => queryOptions({
    queryKey: ["analysis", smiles],
    queryFn: ({ signal }) => fetchAnalysis(smiles, signal),
});
