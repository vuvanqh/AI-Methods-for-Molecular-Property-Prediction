import { createBrowserRouter } from "react-router-dom";
import { AnalysisPage } from "../pages/AnalysisPage";
import { AnalysisResultsPage } from "../pages/AnalysisResultsPage";
import { analysisLoader } from "../api/analysis";

export const routes = createBrowserRouter([
    {
        path: '/',
        element: <AnalysisPage/>,
        children: [
            {
            path: '/:smiles',
            element: <AnalysisResultsPage/>,
            loader: analysisLoader
        }
    ]}
])
