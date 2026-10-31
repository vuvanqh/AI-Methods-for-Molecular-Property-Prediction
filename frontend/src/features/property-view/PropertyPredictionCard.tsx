import type { PropertyPrediction} from "../../types/molecule";
import {useState} from "react";
import {PropertyResultsContainer} from "./PropertyResultsContainer";

export function PropertyPredictionCard({propertyResult}: {propertyResult: PropertyPrediction[], propertyValue: boolean, confidence: number}) {
    const [selectedFamilyIndex, setSelectedPropertyFamily] = useState<number>(0);
    const selectedpropertyFamilty = propertyResult[selectedFamilyIndex];

    return <div className="property-prediction-card">
        <div className="property-prediction-header">
            {selectedpropertyFamilty.property_family}

            <select value={selectedpropertyFamilty.property_family} onChange={(e) => setSelectedPropertyFamily(propertyResult?.findIndex(p => p.property_family=== e.target.value))}>
                {propertyResult.map((property) => (
                    <option key={property.property_family} value={property.property_family}>
                        {property.property_family}
                    </option>
                ))}
            </select>

        </div>
        <PropertyResultsContainer selectedPropertyFamily={selectedpropertyFamilty.prediciton_results} onPropertySelect={()=>{}}/> 

        </div>
}