import { expect, test } from "@playwright/test";
import { analyze, mockApi } from "./fixtures";

test("a shared link or refresh loads that molecule's results", async ({ page }) => {
    const requests = await mockApi(page);
    await page.goto("/CCO");
    await expect(page.getByRole("combobox")).toBeVisible();
    await page.reload(); // the query cache lives in memory, so a reload asks again
    await expect(page.getByRole("combobox")).toBeVisible();
    expect(requests.map(r => r.postDataJSON())).toEqual([{ smiles: "CCO" }, { smiles: "CCO" }]);
});

test("back returns to the landing page and forward to the cached results", async ({ page }) => {
    const requests = await mockApi(page);
    await page.goto("/");
    await analyze(page, "CCO");
    await expect(page.getByRole("combobox")).toBeVisible();

    await page.goBack();
    await expect(page).toHaveURL("/");
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();

    await page.goForward();
    await expect(page).toHaveURL("/CCO");
    await expect(page.getByRole("combobox")).toBeVisible();
    expect(requests).toHaveLength(1); // forward is served from the TanStack cache
});

test("New Analysis returns to an empty landing page", async ({ page }) => {
    await mockApi(page);
    await page.goto("/");
    await analyze(page, "CCO");
    await expect(page.getByRole("combobox")).toBeVisible();

    await page.getByRole("button", { name: "+ New Analysis" }).click();

    await expect(page).toHaveURL("/");
    await expect(page.getByRole("textbox")).toHaveValue("");
    await expect(page.getByRole("button", { name: "Analyze Molecule" })).toBeEnabled();
});
