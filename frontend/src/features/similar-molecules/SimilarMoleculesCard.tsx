import { Metadata } from "../../components/common/Metadata";
import { SimilarMoleculeItem } from "./SimilarMoleculeItem";
import { Section } from "../../components/common/Section";
import type { SimilarCompound } from "../../types/molecule";
import "./SimilarMolecules.css";

type SimilarMoleculesCardProps = {
    similarMolecules: SimilarCompound[]
}

const round2 = (n: number) => Math.round(n * 100) / 100;

export function SimilarMoleculesCard({similarMolecules}: SimilarMoleculesCardProps) {
    if (!similarMolecules || similarMolecules.length===0)
        return null;

    return <Section className="similar-molecules-card card">
        <header className="similar-molecules-header card__header">
            Similar Molecules & Evidence
        </header>

        <div className="table-head">
            <Metadata>Molecule</Metadata>
            <Metadata>Similarity</Metadata>
        </div>

        <ul className="similar-molecules-list" data-animate="stagger">
            {similarMolecules.map(m => (
                <SimilarMoleculeItem key={m.smiles} molecular_formula={m.molecular_formula || m.smiles} score={round2(m.similarity_score)}/>
            ))}
        </ul>

        <Metadata className="similar-molecules-footnote card__footer">
            Higher scores indicate closer structural relatedness.
        </Metadata>
    </Section>
}