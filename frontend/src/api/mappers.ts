// Converts the API response (PredictionResponse) into the props each card expects,
// so components stay independent of the backend's shape.
import type { PredictionResult, PropertiesXaiData, PropertyPrediction, ScientificInterpretation, SimilarCompound, SimilarMolecule } from "../types/molecule";

const round2 = (n: number) => Math.round(n * 100) / 100;

function familyOf(task: string): string {
    if (task.startsWith("CYP")) return "CYP450 inhibition";
    if (task.startsWith("hERG")) return "hERG cardiotoxicity";
    if (task.startsWith("NR-") || task.startsWith("SR-")) return "Toxicity (Tox21)";
    return "Other properties"; // fallback for any new properties that may be added in the future
}

export function toPropertyPredictions({ predictions, probabilities }: PropertiesXaiData): PropertyPrediction[] {
    const families = new Map<string, PredictionResult[]>();
    for (const [name, value] of Object.entries(predictions)) {
        const probability = probabilities[name]; 
        const family = familyOf(name);
        const results = families.get(family) ?? [];
        results.push({ property_name: name, property_value: value, confidence: round2(value ? probability : 1 - probability) });
        families.set(family, results);
    }
    return [...families].map(([property_family, prediciton_results]) => ({ property_family, prediciton_results }));
}

export function toSimilarMolecules(compounds: SimilarCompound[]): SimilarMolecule[] {
    // shortcut: the API has no compound names yet, so the SMILES doubles as the name; use the name once retrieval provides it
    return compounds.map(c => ({ smiles: c.smiles, molecule_name: c.smiles, similarity_score: round2(c.similarity_score) }));
}

export function toScientificInterpretation(summary: string, compounds: SimilarCompound[]): ScientificInterpretation {
    // the evidence behind the summary is the retrieved compounds' sources (deduplicated)
    return { interpretation: summary, sources: [...new Set(compounds.flatMap(c => c.sources))] };
}
