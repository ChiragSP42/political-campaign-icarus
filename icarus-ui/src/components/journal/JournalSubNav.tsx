"use client";

import React from "react";
import Link from "next/link";
import { JOURNAL_SUB_NAV_ITEMS, isActive } from "@/lib/navigation";

export interface JournalSubNavProps {
  currentPath: string;
}

export function JournalSubNav({ currentPath }: JournalSubNavProps) {
  return (
    <nav className="flex flex-row gap-6 border-b border-[var(--border)] px-4 py-2">
      {JOURNAL_SUB_NAV_ITEMS.map((item) => {
        const active = isActive(currentPath, item.activePaths);
        return (
          <Link
            key={item.href}
            href={item.href}
            className={`pb-1 text-sm transition-colors ${
              active
                ? "font-semibold text-[var(--text)] border-b-2 border-[var(--primary)]"
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
