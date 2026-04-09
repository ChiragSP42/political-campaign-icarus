"use client";

import { usePathname } from "next/navigation";
import { JournalSubNav } from "@/components/journal";

export default function JournalLayout({ children }: { children: React.ReactNode }) {
  const currentPath = usePathname();

  return (
    <div className="max-w-7xl mx-auto px-4 md:px-6">
      <JournalSubNav currentPath={currentPath} />
      {children}
    </div>
  );
}
