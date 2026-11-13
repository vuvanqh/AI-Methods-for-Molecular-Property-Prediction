import type { ReactNode } from "react";
import type { MoleculeDescription } from "../../types/molecule";
import { Section } from "../../components/common/Section";
import "./MoleculeOverview.css";
import { Formula } from "../../components/common/MolecularFormula";

const LEGEND = [["C", "Carbon"], ["O", "Oxygen"], ["N", "Nitrogen"], ["H", "Hydrogen"]];

type MoleculeOverviewCardProps = {
    molecule: MoleculeDescription,
    children: ReactNode 
}

export function MoleculeOverviewCard({ molecule, children }: MoleculeOverviewCardProps) {
    return <Section className="molecule-overview card">
        <header className="molecule-overview__header">
            <div>
                <h2 className="molecule-overview__name"><Formula formula={molecule.formula}/></h2>
                <p className="molecule-overview__subtitle mono">{molecule.smiles}</p>
            </div>
            <div className="molecule-overview__chips">
                <span className="chip chip--teal">Browser-rendered 3D</span>
            </div>
        </header>

        <div className="molecule-overview__body">
            <div className="molecule-overview__viewer">
                <div className="molecule-overview__viewer-caption">
                    <strong>Interactive 3D model</strong>
                    <span>Drag to rotate · Scroll to zoom</span>
                </div>
                {children}
                <span className="molecule-overview__credit">3Dmol.js</span>
            </div>

            <div className="molecule-overview__panel">
                <p className="eyebrow">Structural overview</p>
                <h3 className="molecule-overview__title">
                    Molecular profile identified
                    <span className="molecule-overview__status">Ready for interpretation.</span>
                </h3>

                <dl className="molecule-overview__stats">
                    <Stat label="Molecular formula" value={<Formula formula={molecule.formula}/>}/>
                    <Stat label="Molar mass" value={<>{molecule.molar_mass.toFixed(2)} <small>g/mol</small></>}/>
                    <Stat label="Atoms (with H)" value={molecule.atom_count}/>
                    <Stat label="Bonds (with H)" value={molecule.bond_count}/>
                </dl>

                <ul className="molecule-overview__legend">
                    {LEGEND.map(([el, name]) => <li key={el} data-atom={el}>{name}</li>)}
                </ul>
            </div>
        </div>
    </Section>
}

function Stat({ label, value }: { label: string, value: ReactNode }) {
    return <div>
        <dt className="molecule-overview__stat-label">{label}</dt>
        <dd className="molecule-overview__stat-value">{value}</dd>
    </div>
}
