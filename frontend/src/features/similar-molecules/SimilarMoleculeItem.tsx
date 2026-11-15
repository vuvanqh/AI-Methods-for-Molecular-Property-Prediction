import { Formula } from "../../components/common/MolecularFormula"

type SimilarMoleculeItemProps = {
    molecular_formula: string,
    score: number
}

export function SimilarMoleculeItem({molecular_formula, score}: SimilarMoleculeItemProps){
    return <li className="similar-molecule-item">
        <div className="similar-molecule-name">
            <Formula formula={molecular_formula}/>
        </div>
        <div className="similar-molecule-score" data-animate="count" data-value={score}>
            {score}
        </div>
    </li>
}