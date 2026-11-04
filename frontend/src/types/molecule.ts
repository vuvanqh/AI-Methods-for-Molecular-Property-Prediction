export type PropertyPrediction = {
    "property_family": string,
    "prediciton_results": Array<PredictionResult>
}

export type PredictionResult = {
    "property_name": string,
    "property_value": boolean,
    "confidence": number
}

export type SimilarMolecule = {
    "smiles": string,
    "molecule_name": string,
    "similarity_score": number
}

// Response of POST /api/predict-properties (backend/app/src/model.py)
export type PredictionResponse = {
    molecule: MoleculeDescription,
    properties_xai_data: PropertiesXaiData,
    similar_compounds: SimilarCompound[],
    llm_summary: string
}

export type MoleculeDescription = {
    smiles: string,
    formula: string,
    molar_mass: number,
    atom_count: number,
    bond_count: number
}

export type PropertiesXaiData = {
    predictions: Record<string, boolean>,
    probabilities: Record<string, number>,
    atomscores: Record<string, Record<string, number | null>>,
    molblock: string
}

export type SimilarCompound = {
    smiles: string,
    similarity_score: number,
    sources: string[]
}

export type ScientificInterpretation = {
    "interpretation": string,
    "sources": Array<string>
}
