import { Metadata } from "../components/common/Metadata";
import { PropertyPredictionCard } from "../features/property-view/PropertyPredictionCard";
import { ScientificInterpretationCard } from "../features/scientific-interpretation/ScientificInterpretationCard";
import { SimilarMoleculesCard } from "../features/similar-molecules/SimilarMoleculesCard";
import { MoleculeOverviewCard } from "../features/molecule-overview/MoleculeOverviewCard";
import { lazy, Suspense, useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { analysisQuery } from "../api/analysis";
import type { PredictionResponse } from "../types/molecule";
import { toPropertyPredictions, toScientificInterpretation, toSimilarMolecules } from "../api/mappers";
import "../features/molecule-overview/MoleculeOverview.css";

const MoleculeViewer = lazy(() => import("../features/molecule-overview/MoleculeViewer"));

export function AnalysisResultsPage() {
    const { smiles = "" } = useParams()
    const { data, error, isPending, isError } = useQuery(analysisQuery(smiles))

    return <div className="analysis-results">
        <header className="analysis-hero">
            <h2>Understand molecules <span className="gradient-text">beyond the structure.</span></h2>
            <Metadata className="analysis-hero-subtitle">3D molecular visualization, predicted properties, and scientific interpretation in one view.</Metadata>
        </header>
        {isPending ? <div className="molecule-overview--placeholder analysis-loader card" aria-busy="true" data-animate="loader" data-smiles={smiles}>
                <p className="analysis-loader__status" role="status">Loading molecular structure…</p>
            </div>
            : isError ? <p className="molecule-overview molecule-overview--placeholder card" role="alert">{error.message}</p>
            : <AnalysisResults data={data}/>}
    </div>
}

// Owns the selected property, shared by the property card (selection) and the 3D viewer (atom highlighting).
function AnalysisResults({ data }: { data: PredictionResponse }) {
    const [selectedProperty, setSelectedProperty] = useState<string | null>(null)
    const toggleProperty = (name: string) => setSelectedProperty(current => current === name ? null : name)

   return <>
        <MoleculeOverviewCard molecule={data.molecule}>
            <Suspense fallback={null}>
                <MoleculeViewer
                    molblock={data.properties_xai_data.molblock}
                    property={selectedProperty}
                    scores={selectedProperty ? data.properties_xai_data.atomscores[selectedProperty] : undefined}/>
            </Suspense>
        </MoleculeOverviewCard>

        <div className="analysis-page-tier2">
            <div>
                <PropertyPredictionCard
                    key={data.molecule.smiles}
                    propertyResult={toPropertyPredictions(data.properties_xai_data)}
                    selectedProperty={selectedProperty}
                    onPropertySelect={toggleProperty}/>
            </div>
            <div>
                <SimilarMoleculesCard similarMolecules={toSimilarMolecules(data.similar_compounds)}/>
            </div>
        </div>

        <div className="analysis-page-tier3">
            <ScientificInterpretationCard scientificInterpretation={toScientificInterpretation(data.llm_summary, data.similar_compounds)}/>
        </div>
    </>
}