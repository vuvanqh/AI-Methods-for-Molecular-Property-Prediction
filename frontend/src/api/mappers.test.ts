import { describe, expect, it } from "vitest";
import { toPropertyPredictions, toScientificInterpretation } from "./mappers";
import { aspirinResponse } from "../test/fixtures/aspirin";

describe("toPropertyPredictions", () => {
    const families = toPropertyPredictions(aspirinResponse.properties_xai_data);
    const family = (name: string) => families.find(f => f.property_family === name)!.prediciton_results;

    it("groups properties into families, in order of first appearance", () => {
        expect(families.map(f => f.property_family)).toEqual(["CYP450 inhibition", "hERG cardiotoxicity", "Toxicity (Tox21)", "Other properties"]);
        expect(family("CYP450 inhibition").map(p => p.property_name)).toEqual(["CYP1A2_Veith", "CYP2C9_Veith"]);
        expect(family("Toxicity (Tox21)").map(p => p.property_name)).toEqual(["NR-AR", "SR-ARE"]);
        expect(family("Other properties").map(p => p.property_name)).toEqual(["BBBP"]);
    });

    it("reports confidence in the predicted class, rounded to 2 decimals", () => {
        // true at p=0.876 -> 0.88; false at p=0.123 -> 1-0.123 -> 0.88
        expect(family("CYP450 inhibition")).toEqual([
            { property_name: "CYP1A2_Veith", property_value: true, confidence: 0.88 },
            { property_name: "CYP2C9_Veith", property_value: false, confidence: 0.88 },
        ]);
    });

    it("returns no families for no predictions", () => {
        expect(toPropertyPredictions({ predictions: {}, probabilities: {}, atomscores: {}, molblock: "" })).toEqual([]);
    });
});

describe("toScientificInterpretation", () => {
    it("keeps the summary and deduplicates sources in first-seen order", () => {
        expect(toScientificInterpretation("summary", aspirinResponse.similar_compounds)).toEqual({
            interpretation: "summary",
            sources: ["https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL424", "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL25"],
        });
    });

    it("has no sources without compounds", () => {
        expect(toScientificInterpretation("summary", []).sources).toEqual([]);
    });
});
