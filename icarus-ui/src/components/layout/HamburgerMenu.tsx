"use client";

import { useEffect } from "react";
import Link from "next/link";
import { X } from "lucide-react";
import { PRIMARY_NAV_ITEMS, JOURNAL_SUB_NAV_ITEMS, isActive } from "@/lib/navigation";

export interface HamburgerMenuProps {
  isOpen: boolean;
  onClose: () => void;
  currentPath: string;
}

export function getHamburgerLinks(currentPath: string) {
  const showJournalSubs = currentPath.startsWith("/journal");
  return {
    primary: PRIMARY_NAV_ITEMS,
    journal: showJournalSubs ? JOURNAL_SUB_NAV_ITEMS : [],
  };
}

export function HamburgerMenu({ isOpen, onClose, currentPath }: HamburgerMenuProps) {
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const { primary, journal } = getHamburgerLinks(currentPath);

  return (
    <div className="fixed inset-0 z-50">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/30"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div className="absolute right-0 top-0 h-full w-64 bg-white shadow-lg p-6 flex flex-col gap-6">
        <div className="flex justify-end">
          <button
            onClick={onClose}
            className="text-[var(--muted)] hover:text-[var(--text)] cursor-pointer"
            aria-label="Close menu"
          >
            <X size={24} />
          </button>
        </div>

        <nav className="flex flex-col gap-3">
          {primary.map((item) => {
            const active = isActive(currentPath, item.activePaths);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={onClose}
                className={`text-sm transition-colors ${
                  active
                    ? "font-semibold text-[var(--primary)]"
                    : "text-[var(--muted)] hover:text-[var(--text)]"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        {journal.length > 0 && (
          <>
            <hr className="border-[var(--border)]" />
            <nav className="flex flex-col gap-3">
              <span className="text-xs font-medium text-[var(--muted)] uppercase tracking-wide">
                Journal
              </span>
              {journal.map((item) => {
                const active = isActive(currentPath, item.activePaths);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    onClick={onClose}
                    className={`text-sm transition-colors pl-2 ${
                      active
                        ? "font-semibold text-[var(--primary)]"
                        : "text-[var(--muted)] hover:text-[var(--text)]"
                    }`}
                  >
                    {item.label}
                  </Link>
                );
              })}
            </nav>
          </>
        )}
      </div>
    </div>
  );
}
