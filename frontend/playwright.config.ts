import { defineConfig, devices } from "@playwright/test";

const PORT = 4173;

export default defineConfig({
    testDir: "e2e",
    fullyParallel: true,
    // each page runs endless GSAP + WebGL animation; past ~2 browsers they starve each other and page loads time out
    workers: 2,
    forbidOnly: !!process.env.CI,
    retries: process.env.CI ? 1 : 0,
    reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
    use: {
        baseURL: `http://127.0.0.1:${PORT}`,
        trace: "on-first-retry",
    },
    projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
    webServer: {
        // a production build, not the dev server: its first-load dependency optimisation reloads pages mid-test.
        // preview inherits server.proxy (/api -> backend), which the @fullstack smoke relies on
        command: `npx vite build && npx vite preview --host 127.0.0.1 --port ${PORT} --strictPort`,
        timeout: 120_000,
        url: `http://127.0.0.1:${PORT}`,
        reuseExistingServer: !process.env.CI,
    },
});
