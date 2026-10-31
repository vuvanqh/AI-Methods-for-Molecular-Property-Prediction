import type { PredictionResult} from "../../types/molecule";
import { PropertyItem } from "./PropertyItem";

type PropertyContainerProps = {
    selectedPropertyFamily: Array<PredictionResult>
    onPropertySelect?: (propertyName: string) => void
}

export function PropertyResultsContainer({selectedPropertyFamily, onPropertySelect}: PropertyContainerProps) {
    return <div className="property-container">
        <div className="selection-container">
            <ul>
                {selectedPropertyFamily.map( property => 
                    <PropertyItem key={property.property_name} property={property} onPropertySelect={onPropertySelect} />
                )}
            </ul>
        </div>
    </div>
}
