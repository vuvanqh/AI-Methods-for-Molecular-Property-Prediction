export function SmilesInput(){
    return <div className="smiles-input-container">
        <label htmlFor="smiles-input">Enter a molecular structure in SMILES format</label>
        <div>
            <input type="text" id="smiles-input" name="smiles-input" placeholder="e.g. C1=CC=CC=C1" />
            <button className="smiles-input-button">
                Analyze Molecule
            </button>
        </div>
    </div>
}