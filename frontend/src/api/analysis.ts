import type { LoaderFunctionArgs } from "react-router-dom";
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

export function analysisLoader({ params, request }: LoaderFunctionArgs) {
    return { analysis: fetchAnalysis(params.smiles ?? "", request.signal) };
}
