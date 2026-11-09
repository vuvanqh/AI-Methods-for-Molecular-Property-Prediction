import type { Page, Request } from "@playwright/test";
import { aspirinResponse } from "../src/test/fixtures/aspirin";

export { aspirinResponse, ASPIRIN_SMILES } from "../src/test/fixtures/aspirin";

export const API = "**/api/predict-properties";

type Reply = { status?: number, body?: unknown, delayMs?: number };

/** Answers the analysis endpoint with `reply` (the aspirin fixture by default) and records the requests. */
export async function mockApi(page: Page, { status = 200, body = aspirinResponse, delayMs = 0 }: Reply = {}) {
    const requests: Request[] = [];
    await page.route(API, async route => {
        requests.push(route.request());
        if (delayMs) await new Promise(resolve => setTimeout(resolve, delayMs));
        await route.fulfill({ status, contentType: "application/json", body: typeof body === "string" ? body : JSON.stringify(body) });
    });
    return requests;
}

/** Types `smiles` into the landing input and submits with Enter. */
export async function analyze(page: Page, smiles: string) {
    await page.getByRole("textbox").fill(smiles);
    await page.getByRole("textbox").press("Enter");
}
