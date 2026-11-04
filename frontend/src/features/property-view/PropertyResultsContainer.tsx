import type { PredictionResult} from "../../types/molecule";
import { PropertyItem } from "./PropertyItem";

type PropertyContainerProps = {
    selectedPropertyFamily: Array<PredictionResult>
    selectedProperty?: string | null
    onPropertySelect?: (propertyName: string) => void
}

export function PropertyResultsContainer({selectedPropertyFamily, selectedProperty, onPropertySelect}: PropertyContainerProps) {
    return <div className="property-container">
        <div className="selection-container">
            <ul className="property-list" data-animate="stagger">
                {selectedPropertyFamily.map( property =>
                    <PropertyItem key={property.property_name} property={property} selected={property.property_name === selectedProperty} onPropertySelect={onPropertySelect} />
                )}
            </ul>
        </div>
    </div>
}
