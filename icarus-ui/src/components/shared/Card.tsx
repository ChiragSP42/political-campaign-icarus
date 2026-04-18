"use client";

import React from "react";

export interface CardProps {
  children: React.ReactNode;
  header?: React.ReactNode;
  padding?: "sm" | "md" | "lg";
  className?: string;
}

const paddingClasses: Record<NonNullable<CardProps["padding"]>, string> = {
  sm: "p-3",
  md: "p-4",
  lg: "p-6",
};

export function Card({
  children,
  header,
  padding = "md",
  className = "",
}: CardProps) {
  return (
    <div
      className={`bg-white rounded-lg border border-[var(--border)] ${paddingClasses[padding]} ${className}`}
    >
      {header && (
        <div className="mb-3 font-semibold text-[var(--text)]">{header}</div>
      )}
      {children}
    </div>
  );
}
