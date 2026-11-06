type SimilarMoleculeItemProps = {
    name: string,
    score: number
}

export function SimilarMoleculeItem({name, score}: SimilarMoleculeItemProps){
    return <li className="similar-molecule-item">
        <div className="similar-molecule-name">
            {name}
        </div>
        <div className="similar-molecule-score" data-animate="count" data-value={score}>
            {score}
        </div>
    </li>
}