"use client";

import { useRequireAuth } from "@/lib/auth-context";
import { AppShell } from "@/components/layout";

export default function AuthenticatedLayout({ children }: { children: React.ReactNode }) {
  const auth = useRequireAuth();

  if (!auth.authenticated && !auth.email) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p className="text-[var(--muted)]">Loading...</p>
      </div>
    );
  }

  return <AppShell>{children}</AppShell>;
}
