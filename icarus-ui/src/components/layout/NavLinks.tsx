"use client";

import Link from "next/link";
import { PRIMARY_NAV_ITEMS, isActive } from "@/lib/navigation";

export interface NavLinksProps {
  currentPath: string;
  orientation: "horizontal" | "vertical";
  onNavigate?: () => void;
}

export function NavLinks({ currentPath, orientation, onNavigate }: NavLinksProps) {
  return (
    <nav
      className={`flex ${
        orientation === "horizontal" ? "flex-row gap-6" : "flex-col gap-2"
      }`}
    >
      {PRIMARY_NAV_ITEMS.map((item) => {
        const active = isActive(currentPath, item.activePaths);
        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={() => onNavigate?.()}
            className={`text-sm transition-colors ${
              active
                ? "font-semibold underline text-[var(--primary)]"
                : "text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
