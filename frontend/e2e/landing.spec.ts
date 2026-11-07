import { expect, test } from "@playwright/test";
import { mockApi } from "./fixtures";

test.beforeEach(async ({ page }) => {
    await page.goto("/");
});

test("shows the hero, the SMILES input and the analyze button", async ({ page }) => {
    await expect(page.getByRole("heading", { level: 1 })).toContainText("Understand molecules");
    await expect(page.getByLabel("Enter a molecular structure in SMILES format")).toBeVisible();
    await expect(page.getByRole("button", { name: "Analyze Molecule" })).toBeEnabled();
});

for (const blank of ["", "   "]) {
    test(`a blank submit (${JSON.stringify(blank)}) stays on the landing page`, async ({ page }) => {
        const requests = await mockApi(page);
        await page.getByRole("textbox").fill(blank);
        await page.getByRole("button", { name: "Analyze Molecule" }).click();
        await expect(page).toHaveURL("/");
        expect(requests).toHaveLength(0);
    });
}
