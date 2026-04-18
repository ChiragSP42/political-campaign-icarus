"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

export default function Home() {
  const { authenticated, questionnaireCompleted, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (loading) return;
    if (!authenticated) {
      router.replace("/auth");
    } else if (!questionnaireCompleted) {
      router.replace("/questionnaire");
    } else {
      router.replace("/dashboard");
    }
  }, [authenticated, questionnaireCompleted, loading, router]);

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="animate-pulse text-[var(--muted)] text-lg">Loading...</div>
    </div>
  );
}
