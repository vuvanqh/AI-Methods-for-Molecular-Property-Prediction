import { Outlet, useMatch, useNavigate } from "react-router-dom"
import { Navbar } from "../components/layout/Navbar"
import { SmilesInput } from "../features/smiles-input/SmilesInput"
import { useState } from "react"
import { useAnalysisMotion } from "../styles/motion"
import "./AnalysisPage.css"
import { ExemplarySmilesCard } from "../features/smiles-input/ExemplarySmilesCard"

export function AnalysisPage() {
    // derived from the URL, so refresh, shared links and back/forward always show the right view
    const analysisStarted = useMatch("/:smiles") !== null
    const [inputSmiles, setInputSmiles] = useState<string>('')
    const navigate = useNavigate()
    const { exitLanding, exitResults } = useAnalysisMotion({ analysisStarted })

    // play the exit first, then change the route so the next view's entrance can take over
    const analyze = async (smiles: string) => {
        await exitLanding()
        navigate(`/${encodeURIComponent(smiles)}`)
    }

    const newAnalysis = async () => {
        if (!analysisStarted) return
        await exitResults()
        setInputSmiles('')
        navigate('/')
    }

    return <>
        <Navbar onNewAnalysis={newAnalysis}/>
        <div className="molecule-field" aria-hidden="true">
            <div className="molecule-field__item" data-molecule="aspirin"/>
            <div className="molecule-field__item" data-molecule="benzene"/>
            <div className="molecule-field__item" data-molecule="paracetamol"/>
            <div className="molecule-field__item" data-molecule="glycine"/>
            <div className="molecule-field__item" data-molecule="caffeine"/>
            <div className="molecule-field__item" data-molecule="ethanol"/>
            <div className="molecule-field__item" data-molecule="salicylic"/>
            <div className="molecule-field__item" data-molecule="pyridine"/>
            <div className="molecule-field__item molecule-field__item--backdrop" data-molecule="benzene"/>
            <div className="molecule-field__item molecule-field__item--backdrop" data-molecule="caffeine"/>
        </div>
        {!analysisStarted && <header className="landing-hero">
            <h1>Understand molecules <span className="gradient-text">beyond the structure.</span></h1>
        </header>}

        <Outlet/>
        {!analysisStarted && <ExemplarySmilesCard onSelect={setInputSmiles}/>}
        <SmilesInput inputSmiles={inputSmiles} onChange={setInputSmiles} onAnalyze={analyze} disabled={analysisStarted}/>
    </>
}
