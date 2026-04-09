"use client";

import React from "react";

export interface FilterChipProps {
  label: string;
  active: boolean;
  onClick: () => void;
}

export function FilterChip({ label, active, onClick }: FilterChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors cursor-pointer ${
        active
          ? "bg-[var(--primary)] text-white"
          : "border border-[var(--border)] text-[var(--muted)] hover:bg-gray-50"
      }`}
    >
      {label}
    </button>
  );
}
