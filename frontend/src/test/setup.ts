import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import { setReducedMotion } from "./media";
import { queryClient } from "../api/queryClient";

// 3Dmol needs WebGL, which jsdom lacks: every test gets the fake viewer
vi.mock("3dmol", () => import("./mocks/3dmol"));

beforeEach(() => setReducedMotion(true)); // short 0.2 s tweens by default; opt out per test
afterEach(() => {
    cleanup();
    queryClient.clear(); // the app's cache is a module singleton: without this, one test's results leak into the next
    vi.unstubAllGlobals();
});
