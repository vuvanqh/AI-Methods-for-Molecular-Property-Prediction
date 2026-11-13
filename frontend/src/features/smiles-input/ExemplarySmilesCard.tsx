
import { Button } from "../../components/common/Button"

type ExemplarySmilesCardProps = {
    onSelect: (input: string) => void
}

const SMILES = [
    {
        name: "Aspirin",
        smiles: "CC(=O)Oc1ccccc1C(=O)O",
    }, {
        name: "Caffeine",
        smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    }, {
        name: "Ibuprofen",
        smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    }, {
        name: "Glucose",
        smiles: "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    }, {
        name: "Limonene",
        smiles: "CC1=CC[C@H](CC1)C(=C)C"
    }
]

export function ExemplarySmilesCard({onSelect}: ExemplarySmilesCardProps){

    return <div className="smiles-examples">
        <header>
            <h2 className="smiles-examples__label eyebrow">Try an example</h2>
        </header>
        <ul className="smiles-examples__list">
            {SMILES.map(example =>
            <li key={example.smiles} className="smiles-examples__item">
                <Button onClick={()=>onSelect(example.smiles)} type="button" className="smiles-example">
                    <span className="smiles-example__name">{example.name}</span>
                    <span className="smiles-example__smiles">{example.smiles}</span>
                </Button>
            </li>)}
        </ul>
    </div>
}