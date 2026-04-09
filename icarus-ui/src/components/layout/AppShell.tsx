"use client";

import { usePathname } from "next/navigation";
import { TopNav } from "./TopNav";

export interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const currentPath = usePathname();

  return (
    <>
      <TopNav currentPath={currentPath} />
      <main className="pt-16">
        {children}
      </main>
    </>
  );
}
