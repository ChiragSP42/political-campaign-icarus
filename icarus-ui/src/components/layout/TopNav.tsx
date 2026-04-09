"use client";

import { useState } from "react";
import { Menu } from "lucide-react";
import { NavLinks } from "./NavLinks";
import { UserControls } from "./UserControls";
import { HamburgerMenu } from "./HamburgerMenu";

export interface TopNavProps {
  currentPath: string;
}

export function TopNav({ currentPath }: TopNavProps) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <>
      <header className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-[var(--border)]">
        <div className="max-w-7xl mx-auto px-4 md:px-6 h-16 flex items-center justify-between">
          {/* Logo */}
          <span className="text-lg font-bold text-[var(--text)]">WinFlip</span>

          {/* Desktop nav + user controls */}
          <div className="hidden md:flex items-center gap-8">
            <NavLinks currentPath={currentPath} orientation="horizontal" />
            <UserControls />
          </div>

          {/* Mobile hamburger trigger */}
          <button
            className="md:hidden text-[var(--muted)] hover:text-[var(--text)] cursor-pointer"
            onClick={() => setMenuOpen(true)}
            aria-label="Open menu"
          >
            <Menu size={24} />
          </button>
        </div>
      </header>

      <HamburgerMenu
        isOpen={menuOpen}
        onClose={() => setMenuOpen(false)}
        currentPath={currentPath}
      />
    </>
  );
}
