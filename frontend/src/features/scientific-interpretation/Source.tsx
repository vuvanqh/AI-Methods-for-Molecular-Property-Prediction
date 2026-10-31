export function Source({index, source}: {index: number, source: string}) {
    return <div className="source">
        <a href={source} target="_blank" rel="noopener noreferrer">{index}</a>
    </div>
}