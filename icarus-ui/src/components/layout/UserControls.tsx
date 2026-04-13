"use client";

import { useRouter } from "next/navigation";
import { LogOut } from "lucide-react";
import { useAuth } from "@/lib/auth-context";

export function UserControls() {
  const auth = useAuth();
  const router = useRouter();

  const handleLogout = async () => {
    await auth.logout();
    router.push("/auth");
  };

  return (
    <div className="flex items-center gap-3">
      {auth.email && (
        <span className="text-sm text-[var(--muted)]">{auth.email}</span>
      )}
      <button
        onClick={handleLogout}
        className="flex items-center gap-1 text-sm text-[var(--muted)] hover:text-[var(--danger)] transition-colors cursor-pointer"
        aria-label="Logout"
      >
        <LogOut size={16} />
      </button>
    </div>
  );
}
