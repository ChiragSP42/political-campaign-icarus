"use client";

import React from "react";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant: "primary" | "secondary" | "danger";
  size?: "sm" | "md";
}

const variantClasses: Record<ButtonProps["variant"], string> = {
  primary: "bg-[var(--primary)] text-white hover:bg-[var(--primary-dark)]",
  secondary:
    "border border-[var(--border)] bg-transparent text-[var(--text)] hover:bg-gray-50",
  danger: "bg-[var(--danger)] text-white hover:opacity-90",
};

const sizeClasses: Record<NonNullable<ButtonProps["size"]>, string> = {
  sm: "px-3 py-1 text-sm",
  md: "px-4 py-2 text-sm",
};

export function Button({
  variant,
  size = "md",
  className = "",
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={`inline-flex items-center justify-center rounded-md font-medium transition-colors cursor-pointer ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}
