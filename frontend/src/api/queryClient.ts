import { QueryClient } from "@tanstack/react-query";

// One cache for the whole app, created once at module level so re-renders never replace it.
export const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            staleTime: Infinity, // an analysis of a given SMILES never changes, so cached results are never refetched
            retry: false, // a 422 (bad molecule) fails the same way every time; show the error now, not after ~7 s of retries
        },
    },
});
