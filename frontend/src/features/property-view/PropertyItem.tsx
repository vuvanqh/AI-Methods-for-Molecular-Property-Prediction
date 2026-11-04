import type { PredictionResult} from "../../types/molecule";

type PropertyItemProps = {
    property: PredictionResult,
    selected?: boolean,
    onPropertySelect?: (propertyName: string) => void
}

export function PropertyItem({property, selected = false, onPropertySelect}: PropertyItemProps) {
    return  <li className="property-item">
        <button type="button" className="property-button" aria-pressed={selected} onClick={() => onPropertySelect?.(property.property_name)}>
            <span className="property-name">{property.property_name}</span>
            <span className="property-value" data-value={String(property.property_value)}>{property.property_value ? "True" : "False"}</span>
            <span className="property-confidence" data-animate="count" data-value={property.confidence}>Confidence: {property.confidence}</span>
        </button>
    </li>
}