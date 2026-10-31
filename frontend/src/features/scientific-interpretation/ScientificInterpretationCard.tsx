import type {ScientificInterpretation} from "../../types/molecule";
import {Source} from "./Source";

export function ScientificInterpretationCard({scientificInterpretation}: {scientificInterpretation: ScientificInterpretation}) {

    if (!scientificInterpretation.interpretation || scientificInterpretation.interpretation.length === 0) 
        return null;

    return <div className="scientific-interpretation">
        <header className="scientific-interpretation-header">
            <h2>Scientific Interpretation</h2>
        </header>

        <div className="scientific-interpretation-content">
            <div className="explanation">
                <p>{scientificInterpretation.interpretation}</p>
            </div>

            <div className="sources">
                {
                    scientificInterpretation.sources.length!=0 &&
                    scientificInterpretation.sources.map((source, index) => <Source key={source} index={index} source={source}/>)
                }
            </div>
        </div>
    </div>
}