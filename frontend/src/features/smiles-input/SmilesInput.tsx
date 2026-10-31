import {useState} from "react";

export function SmilesInput(){
    const [smiles, setSmiles] = useState("");

    return <div className="smiles-input-container">
        <label htmlFor="smiles-input">Enter a molecular structure in SMILES format</label>

        <div>
            <input type="text" id="smiles-input" name="smiles-input" placeholder="e.g. C1=CC=CC=C1" onChange={(e)=> setSmiles(e.target.value)} value={smiles}/>
            <button className="smiles-input-button">
                Analyze Molecule
            </button>
        </div>
        
    </div>
}