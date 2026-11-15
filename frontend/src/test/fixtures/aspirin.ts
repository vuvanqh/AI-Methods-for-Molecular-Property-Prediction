import type { PredictionResponse } from "../../types/molecule";

export const ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O";

// RDKit 3D conformer (AddHs + ETKDG seed 42 + MMFF): 13 heavy atoms, then 8 hydrogens
export const ASPIRIN_MOLBLOCK = `
     RDKit          3D

 21 21  0  0  0  0  0  0  0  0999 V2000
   -2.5910   -2.2753    0.0671 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7042   -1.1546   -0.3866 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5218   -0.8466   -1.5575 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1623   -0.5293    0.7309 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2481    0.4736    0.3947 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7172    1.7908    0.4314 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.1503    2.8402    0.1348 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4815    2.5743   -0.1844 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.9519    1.2571   -0.2016 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0920    0.1883    0.0957 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5922   -1.2117    0.1078 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.9731   -2.2105    0.4250 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.8783   -1.3038   -0.2826 O   0  0  0  0  0  0  0  0  0  0  0  0
   -2.9743   -2.8095   -0.8070 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.4355   -1.8733    0.6320 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0186   -2.9786    0.6772 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7542    1.9969    0.6810 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2111    3.8656    0.1504 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1577    3.3935   -0.4186 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.9965    1.0779   -0.4464 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.0651   -2.2652   -0.2436 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
  2  4  1  0
  4  5  1  0
  5  6  2  0
  6  7  1  0
  7  8  2  0
  8  9  1  0
  9 10  2  0
 10 11  1  0
 11 12  2  0
 11 13  1  0
 10  5  1  0
  1 14  1  0
  1 15  1  0
  1 16  1  0
  6 17  1  0
  7 18  1  0
  8 19  1  0
  9 20  1  0
 13 21  1  0
M  END
`;

// heavy atoms 0..12 get a score, hydrogens 13..20 are null (as the GNN service returns them)
function atomScores(seed: number): Record<string, number | null> {
    const scores: Record<string, number | null> = {};
    for (let i = 0; i < 21; i++) scores[i] = i < 13 ? Math.round(Math.sin(i + seed) * 1000) / 10000 : null;
    return scores;
}

const PREDICTIONS = {
    CYP1A2_Veith: true,
    CYP2C9_Veith: false,
    hERG: false,
    "NR-AR": true,
    "SR-ARE": false,
    BBBP: true,
};

export const aspirinResponse: PredictionResponse = {
    molecule: { smiles: ASPIRIN_SMILES, formula: "C9H8O4", molar_mass: 180.159, atom_count: 21, bond_count: 21 },
    properties_xai_data: {
        predictions: PREDICTIONS,
        probabilities: { CYP1A2_Veith: 0.876, CYP2C9_Veith: 0.123, hERG: 0.3, "NR-AR": 0.912, "SR-ARE": 0.4, BBBP: 0.7 },
        atomscores: Object.fromEntries(Object.keys(PREDICTIONS).map((name, i) => [name, atomScores(i)])),
        molblock: ASPIRIN_MOLBLOCK,
    },
    similar_compounds: [
        { smiles: "O=C(O)c1ccccc1O", molecular_formula: "C7H6O3", similarity_score: 0.8765, sources: ["https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL424", "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL25"] },
        { smiles: "CC(=O)Nc1ccc(O)cc1", molecular_formula: "C8H9NO2", similarity_score: 0.4321, sources: ["https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL25"] },
    ],
    llm_summary: "Aspirin is predicted to inhibit CYP1A2 and to be inactive at hERG.",
};
