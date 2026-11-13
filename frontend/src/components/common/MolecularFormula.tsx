export function Formula({ formula }: { formula: string }) {
    return <>{formula.split(/(\d+)/).map((part, i) => i % 2 ? <sub key={i}>{part}</sub> : part)}</>
}
