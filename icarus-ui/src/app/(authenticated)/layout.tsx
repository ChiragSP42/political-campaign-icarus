"use client";

import { useRequireAuth } from "@/lib/auth-context";
import { AppShell } from "@/components/layout";
import { Loader2 } from "lucide-react";

export default function AuthenticatedLayout({ children }: { children: React.ReactNode }) {
  const auth = useRequireAuth();

  if (auth.loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 size={32} className="animate-spin text-[var(--primary)]" />
      </div>
    );
  }

  if (!auth.authenticated) {
    return null;
  }

  return <AppShell>{children}</AppShell>;
}
