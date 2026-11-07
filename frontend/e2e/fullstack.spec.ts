// Real stack, local only: needs the Docker DB, the GNN service and the backend app on :8001 (Vite proxies /api).
// Run with `npm run test:e2e:fullstack`. Checks the response's shape, not its values, since models change.
import { expect, test } from "@playwright/test";
import { analyze, ASPIRIN_SMILES } from "./fixtures";

test("aspirin gets real predictions @fullstack", async ({ page }) => {
    test.setTimeout(90_000);
    await page.goto("/");
    await analyze(page, ASPIRIN_SMILES);

    await expect(page.getByRole("combobox")).toBeVisible({ timeout: 60_000 });
    await expect(page.getByRole("alert")).toHaveCount(0);
    await expect(page.locator(".property-item").first()).toBeVisible();
    await expect(page.locator(".similar-molecule-item").first()).toBeVisible();
    await expect(page.locator(".molecule-overview__canvas canvas")).toHaveCount(1);
});
