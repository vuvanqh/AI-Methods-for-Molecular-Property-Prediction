import { Form } from "react-router-dom";
import { Button } from "../../components/common/Button";
import { Section } from "../../components/common/Section";
import "./SmilesInput.css";

type SmilesInputProps = {
    inputSmiles: string,
    onChange: (input:string)=>void,
    onAnalyze: (smiles: string)=>void,
    disabled: boolean,
}

export function SmilesInput({inputSmiles, onChange, onAnalyze, disabled}: SmilesInputProps){
    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (inputSmiles.trim()) {
            onAnalyze(inputSmiles.trim());
        }
    };
    return <Section className="smiles-input-container card">
        <label htmlFor="smiles-input" className="smiles-input-label">Enter a molecular structure in SMILES format</label>

        <Form onSubmit={handleSubmit} className="smiles-input-form">
            <input type="text" id="smiles-input" name="smiles-input" placeholder="e.g. C1=CC=CC=C1" onChange={(e)=> onChange(e.target.value)} value={inputSmiles} spellCheck={false} autoComplete="off" disabled={disabled}/>
            <Button className="smiles-input-button btn btn--primary" disabled={disabled} type="submit">
                Analyze Molecule
            </Button>
        </Form>

    </Section>
}