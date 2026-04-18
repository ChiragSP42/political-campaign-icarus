"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { OFFICE_OPTIONS, STATEWIDE_OFFICES, BACKGROUND_QUESTIONS, ARCHETYPE_QUESTIONS } from "@/lib/constants";
import { ChevronLeft, ChevronRight, Send, Loader2 } from "lucide-react";

const TOTAL_STEPS = 3;

export default function QuestionnairePage() {
  const auth = useAuth();
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [polling, setPolling] = useState(false);
  const [pollMsg, setPollMsg] = useState("");
  const [showSlowNote, setShowSlowNote] = useState(false);

  const set = (key: string, val: any) => setFormData((p) => ({ ...p, [key]: val }));

  const isStatewide = STATEWIDE_OFFICES.has(formData.office_position || "");

  async function handleSubmit() {
    setLoading(true); setError("");
    try {
      // Delete old insights so polling waits for the new ones
      await fetch(`/api/insights?email=${encodeURIComponent(auth.email!)}`, { method: "DELETE" });

      const res = await fetch("/api/questionnaire/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: auth.email, answers: formData }),
      });
      const data = await res.json();
      if (!data.success && !data.message?.includes?.("success")) throw new Error(data.message || "Failed to save");
      setPolling(true);
      pollForInsights();
    } catch (err: any) {
      setError(err.message);
      setLoading(false);
    }
  }

  async function pollForInsights() {
    const msgs = [
      "🔍 Analyzing historical election data...",
      "📈 Calculating Win Gap scenarios...",
      "🗺️ Processing precinct-level targeting...",
      "💡 Generating strategic recommendations...",
      "🎯 Finalizing your campaign insights...",
    ];
    const start = Date.now();
    const initialWait = 300000; // 5 minutes before showing "taking longer" message
    let i = 0;

    while (Date.now() - start < initialWait) {
      setPollMsg(msgs[i % msgs.length]);
      i++;
      try {
        const res = await fetch(`/api/insights?email=${encodeURIComponent(auth.email!)}`);
        const data = await res.json();
        if (data.exists && data.content) {
          auth.setUser(auth.email!, true);
          router.push("/dashboard");
          return;
        }
      } catch { /* keep polling */ }
      await new Promise((r) => setTimeout(r, 5000));
    }
    // Insights not ready after 4 min — show a friendly nudge instead of an error
    setPolling(false);
    setLoading(false);
    setShowSlowNote(true);
  }

  if (showSlowNote) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-lg border border-[var(--border)] p-8 max-w-lg w-full text-center">
          <span className="text-4xl mb-4 block">🗺️</span>
          <h2 className="text-xl font-semibold mb-2">Crunching a lot of data!</h2>
          <p className="text-[var(--muted)] mb-4">
            It looks like the area you're running in is large, so the analytics are taking a bit longer than usual. No worries — the AI is still working on it.
          </p>
          <p className="text-sm text-[var(--muted)] mb-6">
            Let's head over to the dashboard. Your insights will appear automatically once they're ready.
          </p>
          <button onClick={() => { auth.setUser(auth.email!, true); router.push("/dashboard"); }}
            className="px-6 py-2.5 bg-[var(--primary)] hover:bg-[var(--primary-dark)] text-white rounded-lg text-sm font-medium transition">
            Go to Dashboard →
          </button>
        </div>
      </div>
    );
  }

  if (polling) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-lg border border-[var(--border)] p-8 max-w-lg w-full text-center">
          <Loader2 size={48} className="animate-spin text-[var(--primary)] mx-auto mb-4" />
          <h2 className="text-xl font-semibold mb-2">Generating Your Campaign Insights</h2>
          <p className="text-[var(--muted)] mb-4">This typically takes 5-7 minutes.</p>
          <p className="text-sm font-medium text-[var(--primary)]">{pollMsg}</p>
          <button onClick={() => { router.push("/dashboard"); }}
            className="mt-6 text-sm text-[var(--muted)] hover:text-[var(--text)] underline">
            Go to dashboard anyway →
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-lg border border-[var(--border)] w-full max-w-2xl">
        {/* Progress */}
        <div className="p-6 border-b border-[var(--border)]">
          <h1 className="text-2xl font-bold mb-1">📋 Candidate Intake Questionnaire</h1>
          <p className="text-[var(--muted)] text-sm">Step {step} of {TOTAL_STEPS}</p>
          <div className="mt-3 h-2 bg-gray-100 rounded-full overflow-hidden">
            <div className="h-full bg-[var(--primary)] rounded-full transition-all duration-300"
              style={{ width: `${(step / TOTAL_STEPS) * 100}%` }} />
          </div>
        </div>

        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {error && <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>}

          {step === 1 && (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">Basic Information</h2>
              <div>
                <label className="block text-sm font-medium mb-1">Full Name *</label>
                <input value={formData.fullName || ""} onChange={(e) => set("fullName", e.target.value)}
                  className="w-full px-4 py-2.5 border border-[var(--border)] rounded-lg focus:ring-2 focus:ring-[var(--primary)] outline-none" placeholder="Your full name" />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Office Running For *</label>
                <select value={formData.office_position || ""} onChange={(e) => set("office_position", e.target.value)}
                  className="w-full px-4 py-2.5 border border-[var(--border)] rounded-lg focus:ring-2 focus:ring-[var(--primary)] outline-none bg-white">
                  <option value="">Select office...</option>
                  {OFFICE_OPTIONS.map((o) => <option key={o} value={o}>{o.replace(/_/g, " ")}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">District *</label>
                {isStatewide ? (
                  <input value="Statewide" disabled className="w-full px-4 py-2.5 border border-[var(--border)] rounded-lg bg-gray-50" />
                ) : (
                  <select value={formData.district_name || ""} onChange={(e) => set("district_name", e.target.value)}
                    className="w-full px-4 py-2.5 border border-[var(--border)] rounded-lg focus:ring-2 focus:ring-[var(--primary)] outline-none bg-white">
                    <option value="">Select district...</option>
                    {Array.from({ length: 100 }, (_, i) => (
                      <option key={i + 1} value={`District_${i + 1}`}>District {i + 1}</option>
                    ))}
                  </select>
                )}
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-3">
              <h2 className="text-lg font-semibold">Background & Profile</h2>
              <p className="text-sm text-[var(--muted)]">Tell us about your background and credibility anchors</p>
              {Object.entries(BACKGROUND_QUESTIONS).map(([key, question]) => (
                <label key={key} className="flex items-start gap-3 p-3 rounded-lg hover:bg-gray-50 cursor-pointer transition">
                  <input type="checkbox" checked={!!formData[question]}
                    onChange={(e) => set(question, e.target.checked)}
                    className="mt-0.5 w-4 h-4 rounded border-gray-300 text-[var(--primary)] focus:ring-[var(--primary)]" />
                  <span className="text-sm">{question}</span>
                </label>
              ))}
            </div>
          )}

          {step === 3 && (
            <div className="space-y-6">
              <h2 className="text-lg font-semibold">Communication Style & Archetype</h2>
              {Object.entries(ARCHETYPE_QUESTIONS).map(([key, { question, options }]) => (
                <div key={key}>
                  <p className="text-sm font-medium mb-2">{question}</p>
                  <div className="space-y-1.5">
                    {options.map((opt) => (
                      <label key={opt} className={`flex items-center gap-3 p-2.5 rounded-lg cursor-pointer transition text-sm border
                        ${formData[question] === opt ? "border-[var(--primary)] bg-indigo-50" : "border-transparent hover:bg-gray-50"}`}>
                        <input type="radio" name={key} checked={formData[question] === opt}
                          onChange={() => set(question, opt)}
                          className="w-4 h-4 text-[var(--primary)] focus:ring-[var(--primary)]" />
                        <span>{opt}</span>
                      </label>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="p-6 border-t border-[var(--border)] flex justify-between">
          <button onClick={() => setStep((s) => s - 1)} disabled={step === 1}
            className="flex items-center gap-1 px-4 py-2 text-sm font-medium text-[var(--muted)] hover:text-[var(--text)] disabled:opacity-30 transition">
            <ChevronLeft size={16} /> Previous
          </button>
          {step < TOTAL_STEPS ? (
            <button onClick={() => {
              if (step === 1 && (!formData.fullName || !formData.office_position)) {
                setError("Please fill in all required fields"); return;
              }
              setError(""); setStep((s) => s + 1);
            }}
              className="flex items-center gap-1 px-6 py-2 bg-[var(--primary)] hover:bg-[var(--primary-dark)] text-white rounded-lg text-sm font-medium transition">
              Next <ChevronRight size={16} />
            </button>
          ) : (
            <button onClick={handleSubmit} disabled={loading}
              className="flex items-center gap-2 px-6 py-2 bg-[var(--accent)] hover:bg-green-600 text-white rounded-lg text-sm font-medium transition disabled:opacity-50">
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
              {loading ? "Submitting..." : "Submit"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
