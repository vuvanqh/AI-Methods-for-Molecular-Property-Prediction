import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Button } from "./Button";
import { Metadata } from "./Metadata";
import { Section } from "./Section";

describe("Button", () => {
    it("forwards native button props and renders its children", async () => {
        const onClick = vi.fn();
        render(<Button type="submit" className="btn" onClick={onClick}>Go</Button>);
        const button = screen.getByRole("button", { name: "Go" });
        expect(button).toHaveAttribute("type", "submit");
        expect(button).toHaveClass("btn");
        await userEvent.click(button);
        expect(onClick).toHaveBeenCalledOnce();
    });

    it("does not fire clicks when disabled", async () => {
        const onClick = vi.fn();
        render(<Button disabled onClick={onClick}>Go</Button>);
        expect(screen.getByRole("button")).toBeDisabled();
        await userEvent.click(screen.getByRole("button"));
        expect(onClick).not.toHaveBeenCalled();
    });
});

describe("Metadata", () => {
    it("adds the given class to the base class", () => {
        render(<Metadata className="footnote">note</Metadata>);
        expect(screen.getByText("note")).toHaveAttribute("class", "metadata footnote");
    });

    it("has only the base class (no trailing space) without className", () => {
        render(<Metadata>note</Metadata>);
        expect(screen.getByText("note")).toHaveAttribute("class", "metadata");
    });
});

describe("Section", () => {
    it("renders a <section> with the given class", () => {
        const { container } = render(<Section className="card">body</Section>);
        const section = container.querySelector("section");
        expect(section).toHaveClass("card");
        expect(section).toHaveTextContent("body");
    });
});
