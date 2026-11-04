import type { PropertyPrediction} from "../../types/molecule";
import {useState} from "react";
import {PropertyResultsContainer} from "./PropertyResultsContainer";
import { Section } from "../../components/common/Section";
import "./PropertyView.css";

type PropertyPredictionCardProps = {
    propertyResult: PropertyPrediction[],
    selectedProperty?: string | null,
    onPropertySelect?: (propertyName: string) => void
}

export function PropertyPredictionCard({propertyResult, selectedProperty, onPropertySelect}: PropertyPredictionCardProps) {
    const [selectedFamilyIndex, setSelectedPropertyFamilyIndex] = useState<number>(0);
    const selectedPropertyFamily = propertyResult[selectedFamilyIndex];

    if (!propertyResult || propertyResult.length===0)
        return <p className="empty-state">No predictions available</p>;

    return <Section className="property-prediction-card card">
            <div className="property-prediction-header card__header">
                {selectedPropertyFamily.property_family}

                <select className="property-family-select" value={selectedPropertyFamily.property_family} onChange={(e) => setSelectedPropertyFamilyIndex(propertyResult?.findIndex(p => p.property_family=== e.target.value))}>
                    {propertyResult.map((property) => (
                        <option key={property.property_family} value={property.property_family}>
                            {property.property_family}
                        </option>
                    ))}
                </select>
            </div>
            
            <PropertyResultsContainer selectedPropertyFamily={selectedPropertyFamily.prediciton_results} selectedProperty={selectedProperty} onPropertySelect={onPropertySelect}/>
        </Section>
}