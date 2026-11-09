import type { ReactNode } from "react";
import { render } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { vi } from "vitest";
import { routes } from "../routes/routes";
import { queryClient } from "../api/queryClient";

/** Renders `ui` as the only route, for components that need router context (e.g. <Form>). */
export function renderWithRouter(ui: ReactNode) {
    const router = createMemoryRouter([{ path: "/", element: ui }]);
    return { router, ...render(<RouterProvider router={router}/>) };
}

/** Renders the real app (query cache + routes) at `url`, in memory instead of the browser history. */
export function renderApp(url = "/") {
    const router = createMemoryRouter(routes.routes, { initialEntries: [url] });
    return { router, ...render(<QueryClientProvider client={queryClient}><RouterProvider router={router}/></QueryClientProvider>) };
}

/** Replaces fetch with one that answers every call with `body` and `status`. */
export function mockFetch(body: unknown, status = 200) {
    const fetch = vi.fn(async (_url: string, _init?: RequestInit) =>
        new Response(typeof body === "string" ? body : JSON.stringify(body), { status }));
    vi.stubGlobal("fetch", fetch);
    return fetch;
}

/** Replaces fetch with one that never answers, but rejects like the real one when its signal aborts. */
export function mockPendingFetch() {
    const fetch = vi.fn((_url: string, init?: RequestInit) => new Promise<Response>((_, reject) =>
        init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")))));
    vi.stubGlobal("fetch", fetch);
    return fetch;
}

/** The parsed JSON body of the n-th fetch call. */
export const requestBody = (fetch: ReturnType<typeof vi.fn>, n = 0) => JSON.parse((fetch.mock.calls[n][1] as RequestInit).body as string);
