import type { PredictionResult} from "../../types/molecule";

type PropertyItemProps = {
    property: PredictionResult,
    onPropertySelect?: (propertyName: string) => void
}

export function PropertyItem({property, onPropertySelect}: PropertyItemProps) {
    return  <li className="property-item">
        <button className="property-button" onClick={() => onPropertySelect?.(property.property_name)}>
            <span className="property-name">{property.property_name}</span>
            <span className="property-value">{property.property_value ? "True" : "False"}</span>
            <span className="property-confidence">Confidence: {property.confidence}</span>
        </button>
    </li>
}