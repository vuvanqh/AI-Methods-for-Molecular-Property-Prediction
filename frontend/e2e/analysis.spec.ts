import { expect, test } from "@playwright/test";
import { analyze, ASPIRIN_SMILES, aspirinResponse, mockApi } from "./fixtures";

test("analyzes a molecule end to end", async ({ page }) => {
    const requests = await mockApi(page, { delayMs: 500 }); // long enough to see the loading state
    await page.goto("/");

    await analyze(page, `  ${ASPIRIN_SMILES}  `);

    await expect(page).toHaveURL(`/${encodeURIComponent(ASPIRIN_SMILES)}`);
    await expect(page.getByText("Loading molecular structure…")).toBeVisible();
    expect(requests[0].postDataJSON()).toEqual({ smiles: ASPIRIN_SMILES });

    await expect(page.getByRole("heading", { level: 2, name: "C9H8O4" })).toBeVisible();
    await expect(page.getByRole("combobox")).toHaveValue("CYP450 inhibition");
    await expect(page.getByText("Similar Molecules & Evidence")).toBeVisible();
    await expect(page.getByText(aspirinResponse.llm_summary)).toBeVisible();
    await expect(page.locator(".molecule-overview__canvas canvas")).toHaveCount(1); // real 3Dmol (WebGL)
});

test.describe("results page", () => {
    test.beforeEach(async ({ page }) => {
        await mockApi(page);
        await page.goto(`/${encodeURIComponent(ASPIRIN_SMILES)}`);
        await expect(page.getByRole("combobox")).toBeVisible();
    });

    test("selecting a property shows its atom attribution, selecting it again hides it", async ({ page }) => {
        const property = page.getByRole("button", { name: /CYP1A2_Veith/ });
        await property.click();
        await expect(property).toHaveAttribute("aria-pressed", "true");
        await expect(page.getByText("Atom attribution · CYP1A2_Veith")).toBeVisible();
        await expect(page.getByRole("img", { name: /attribution to CYP1A2_Veith/ })).toBeVisible();
        await expect(page.locator(".molecule-overview__overlay g")).toHaveCount(13);

        await property.click();
        await expect(property).toHaveAttribute("aria-pressed", "false");
        await expect(page.getByText(/Atom attribution/)).toHaveCount(0);
    });

    test("the family select swaps the property rows", async ({ page }) => {
        await page.getByRole("combobox").selectOption("Toxicity (Tox21)");
        await expect(page.locator(".property-name")).toHaveText(["NR-AR", "SR-ARE"]);
    });

    test("the rotation toggle pauses and resumes", async ({ page }) => {
        await page.getByRole("button", { name: "Pause rotation" }).click();
        await page.getByRole("button", { name: "Resume rotation" }).click();
        await expect(page.getByRole("button", { name: "Pause rotation" })).toHaveAttribute("aria-pressed", "true");
    });

    test("source links open in a new tab", async ({ page }) => {
        const links = page.locator(".sources a");
        await expect(links).toHaveCount(2);
        for (const link of await links.all()) {
            await expect(link).toHaveAttribute("target", "_blank");
            await expect(link).toHaveAttribute("rel", "noopener noreferrer");
        }
    });
});

test("works with reduced motion, and the model starts paused", async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await mockApi(page);
    await page.goto("/");
    await analyze(page, ASPIRIN_SMILES);
    await expect(page.getByRole("combobox")).toBeVisible();
    await expect(page.getByRole("button", { name: "Resume rotation" })).toBeVisible();
});
