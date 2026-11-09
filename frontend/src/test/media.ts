// jsdom has no matchMedia; this stub answers only prefers-reduced-motion, which is all the app asks
let reduced = true;

export function setReducedMotion(value: boolean) {
    reduced = value;
}

Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: (query: string) => ({
        matches: query.includes("prefers-reduced-motion: reduce") && reduced,
        media: query,
        onchange: null,
        addEventListener: () => {},
        removeEventListener: () => {},
        addListener: () => {},
        removeListener: () => {},
        dispatchEvent: () => false,
    }),
});
