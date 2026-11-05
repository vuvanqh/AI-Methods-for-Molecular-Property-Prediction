import "./Navbar.css";

export function Navbar({ onNewAnalysis }: { onNewAnalysis: () => void }){
    return <nav className="navbar">
        <header className="navbar-title">
            Molecular Property Analysis
        </header>
        <div className="navbar-actions">
            <button type="button" className="navbar-button btn btn--ghost" onClick={onNewAnalysis}>
                + New Analysis
            </button>
        </div>
    </nav>
}