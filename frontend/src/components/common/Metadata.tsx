import type { ReactNode } from "react";

export function Metadata({children, className}: {className?: string, children: ReactNode}) {
    const fullClassName = `metadata ${className??""}`.trim();
    return <p className={fullClassName}>
        {children}
    </p>
}