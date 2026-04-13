"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { LogIn, UserPlus, Mail, Lock, ShieldCheck, Loader2 } from "lucide-react";

type Tab = "signin" | "signup";
type SignupStep = "form" | "verify";

export default function AuthPage() {
  const [tab, setTab] = useState<Tab>("signin");
  const [signupStep, setSignupStep] = useState<SignupStep>("form");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [signupEmail, setSignupEmail] = useState("");
  const { setUser, authenticated, questionnaireCompleted, loading: authLoading } = useAuth();
  const router = useRouter();

  // Redirect if already authenticated
  useEffect(() => {
    if (!authLoading && authenticated) {
      router.replace(questionnaireCompleted ? "/dashboard" : "/questionnaire");
    }
  }, [authLoading, authenticated, questionnaireCompleted, router]);

  if (authLoading || authenticated) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 size={32} className="animate-spin text-[var(--primary)]" />
      </div>
    );
  }

  async function handleSignIn(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(""); setLoading(true);
    const fd = new FormData(e.currentTarget);
    try {
      const res = await fetch("/api/auth/signin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: fd.get("email"), password: fd.get("password") }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.message);
      setUser(data.email, data.questionnaireCompleted);
      router.push(data.questionnaireCompleted ? "/dashboard" : "/questionnaire");
    } catch (err: any) {
      setError(err.message);
    } finally { setLoading(false); }
  }

  async function handleSignUp(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(""); setLoading(true);
    const fd = new FormData(e.currentTarget);
    const password = fd.get("password") as string;
    if (password !== fd.get("confirm")) { setError("Passwords do not match"); setLoading(false); return; }
    const email = fd.get("email") as string;
    try {
      const res = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.message);
      setSignupEmail(email);
      setSignupStep("verify");
      setSuccess(data.message);
    } catch (err: any) {
      setError(err.message);
    } finally { setLoading(false); }
  }

  async function handleVerify(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(""); setLoading(true);
    const fd = new FormData(e.currentTarget);
    try {
      const res = await fetch("/api/auth/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: signupEmail, code: fd.get("code") }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.message);
      setSuccess(data.message);
      setSignupStep("form");
      setTab("signin");
    } catch (err: any) {
      setError(err.message);
    } finally { setLoading(false); }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-[var(--primary)] mb-2">🎭 Project Icarus</h1>
          <p className="text-[var(--muted)]">Campaign Strategy AI</p>
        </div>

        <div className="bg-white rounded-2xl shadow-lg border border-[var(--border)] overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b border-[var(--border)]">
            {(["signin", "signup"] as Tab[]).map((t) => (
              <button key={t} onClick={() => { setTab(t); setError(""); setSuccess(""); }}
                className={`flex-1 py-3 px-4 text-sm font-medium transition-colors flex items-center justify-center gap-2
                  ${tab === t ? "text-[var(--primary)] border-b-2 border-[var(--primary)] bg-indigo-50/50" : "text-[var(--muted)] hover:text-[var(--text)]"}`}>
                {t === "signin" ? <><LogIn size={16} /> Sign In</> : <><UserPlus size={16} /> Sign Up</>}
              </button>
            ))}
          </div>

          <div className="p-6">
            {error && <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>}
            {success && <div className="mb-4 p-3 bg-green-50 text-green-700 rounded-lg text-sm">{success}</div>}

            {tab === "signin" && (
              <form onSubmit={handleSignIn} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Email</label>
                  <div className="relative">
                    <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                    <input name="email" type="email" required placeholder="you@example.com"
                      className="w-full pl-10 pr-4 py-2.5 border border-[var(--border)] rounded-lg focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent outline-none transition" />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Password</label>
                  <div className="relative">
                    <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                    <input name="password" type="password" required placeholder="Enter your password"
                      className="w-full pl-10 pr-4 py-2.5 border border-[var(--border)] rounded-lg focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent outline-none transition" />
                  </div>
                </div>
                <button type="submit" disabled={loading}
                  className="w-full py-2.5 bg-[var(--primary)] hover:bg-[var(--primary-dark)] text-white rounded-lg font-medium transition disabled:opacity-50">
                  {loading ? "Signing in..." : "Sign In"}
                </button>
              </form>
            )}

            {tab === "signup" && signupStep === "form" && (
              <form onSubmit={handleSignUp} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Email</label>
                  <div className="relative">
                    <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                    <input name="email" type="email" required placeholder="you@example.com"
                      className="w-full pl-10 pr-4 py-2.5 border border-[var(--border)] rounded-lg focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent outline-none transition" />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Password</label>
                  <div className="relative">
                    <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                    <input name="password" type="password" required placeholder="8+ chars, upper, lower, numbers"
                      className="w-full pl-10 pr-4 py-2.5 border border-[var(--border)] rounded-lg focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent outline-none transition" />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Confirm Password</label>
                  <div className="relative">
                    <ShieldCheck size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                    <input name="confirm" type="password" required placeholder="Confirm password"
                      className="w-full pl-10 pr-4 py-2.5 border border-[var(--border)] rounded-lg focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent outline-none transition" />
                  </div>
                </div>
                <button type="submit" disabled={loading}
                  className="w-full py-2.5 bg-[var(--primary)] hover:bg-[var(--primary-dark)] text-white rounded-lg font-medium transition disabled:opacity-50">
                  {loading ? "Creating account..." : "Sign Up"}
                </button>
              </form>
            )}

            {tab === "signup" && signupStep === "verify" && (
              <form onSubmit={handleVerify} className="space-y-4">
                <p className="text-sm text-[var(--muted)]">Enter the 6-digit code sent to <span className="font-medium text-[var(--text)]">{signupEmail}</span></p>
                <input name="code" type="text" required placeholder="123456" maxLength={6}
                  className="w-full px-4 py-2.5 border border-[var(--border)] rounded-lg text-center text-2xl tracking-widest focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent outline-none" />
                <button type="submit" disabled={loading}
                  className="w-full py-2.5 bg-[var(--primary)] hover:bg-[var(--primary-dark)] text-white rounded-lg font-medium transition disabled:opacity-50">
                  {loading ? "Verifying..." : "Verify Email"}
                </button>
              </form>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
