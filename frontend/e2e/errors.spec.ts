import { expect, test } from "@playwright/test";
import { API, mockApi } from "./fixtures";

const cases = [
    { status: 422, body: { message: "The molecule could not be processed." }, message: "The molecule could not be processed." },
    { status: 503, body: { message: "GNN service unavailable." }, message: "GNN service unavailable." },
    { status: 500, body: "Internal Server Error", message: "Analysis failed (500)." },
];

for (const { status, body, message } of cases) {
    test(`a ${status} response shows its message`, async ({ page }) => {
        await mockApi(page, { status, body });
        await page.goto("/C1CC");
        await expect(page.getByRole("alert")).toHaveText(message);
        await expect(page.getByRole("combobox")).toHaveCount(0);
    });
}

test("a network failure shows an alert", async ({ page }) => {
    await page.route(API, route => route.abort("connectionfailed"));
    await page.goto("/CCO");
    await expect(page.getByRole("alert")).toBeVisible();
});
