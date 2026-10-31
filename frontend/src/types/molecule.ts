export type MolecularAnalysisResponse = {
    "property_prediction": PropertyPrediction[],
    "similar_molecules": SimilarMolecules,
    "scientific_interpretation": ScientificInterpretation
}

export type PropertyPrediction = {
    "property_family": string,
    "prediciton_results": Array<PredictionResult>
}

export type PredictionResult = {
    "property_name": string,
    "property_value": boolean,
    "confidence": number
}

export type SimilarMolecules = {
    "molecules": Array<SimilarMolecule>
}

type SimilarMolecule = {
    "smiles": string,
    "molecule_name": string,
    "similarity_score": number
}

export type ScientificInterpretation = {
    "interpretation": string,
    "sources": Array<string>
}