"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useAuth } from "@/lib/auth-context";
import ReactMarkdown from "react-markdown";
import {
  Send, Trash2, Download, RefreshCw, FileText, MessageSquare, Loader2,
} from "lucide-react";

interface ChatMsg {
  role: "user" | "assistant";
  content: [{ text: string }];
}

export default function DashboardPage() {
  const auth = useAuth();
  const [insights, setInsights] = useState<string | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(true);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const loadInsights = useCallback(async () => {
    if (!auth.email) return;
    setInsightsLoading(true);
    try {
      const res = await fetch(`/api/insights?email=${encodeURIComponent(auth.email)}`);
      const data = await res.json();
      if (data.exists) setInsights(data.content);
      else setInsights(null);
    } catch { setInsights(null); }
    setInsightsLoading(false);
  }, [auth.email]);

  useEffect(() => { loadInsights(); }, [loadInsights]);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function sendMessage() {
    if (!input.trim() || sending) return;
    const userMsg: ChatMsg = { role: "user", content: [{ text: input }] };
    setMessages((m) => [...m, userMsg]);
    const query = input;
    setInput("");
    setSending(true);

    try {
      const history = [...messages, userMsg].map((m) => ({ role: m.role, content: m.content }));
      const res = await fetch("/api/chat/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: auth.email, query, conversation_history: history }),
      });
      const data = await res.json();

      if (data.status === "COMPLETED") {
        // Poll for response
        while (true) {
          const checkRes = await fetch(`/api/chat/check?email=${encodeURIComponent(auth.email!)}`);
          const checkData = await checkRes.json();
          if (checkData.status === "COMPLETED") {
            setMessages((m) => [...m, { role: "assistant", content: [{ text: checkData.message || "No response" }] }]);
            break;
          } else if (checkData.status === "FAILED") {
            setMessages((m) => [...m, { role: "assistant", content: [{ text: "Sorry, something went wrong. Please try again." }] }]);
            break;
          }
          await new Promise((r) => setTimeout(r, 2000));
        }
      } else {
        setMessages((m) => [...m, { role: "assistant", content: [{ text: data.message || "Error occurred" }] }]);
      }
    } catch (err: any) {
      setMessages((m) => [...m, { role: "assistant", content: [{ text: `Error: ${err.message}` }] }]);
    }
    setSending(false);
  }

  function downloadInsights() {
    if (!insights) return;
    const blob = new Blob([insights], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${auth.email}_campaign_insights.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="flex flex-col" style={{ height: "calc(100vh - 4rem)" }}>
      {/* Main content — stacks vertically on mobile, side-by-side on desktop */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Insights panel */}
        <div className="w-full md:w-1/2 border-b md:border-b-0 md:border-r border-[var(--border)] flex flex-col bg-white">
          <div className="px-5 py-3 border-b border-[var(--border)] flex items-center justify-between shrink-0">
            <h2 className="font-semibold flex items-center gap-2"><FileText size={18} /> Campaign Insights</h2>
            <div className="flex gap-2">
              {insights && (
                <button onClick={downloadInsights}
                  className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-[var(--muted)] hover:text-[var(--text)] hover:bg-gray-100 rounded-lg transition">
                  <Download size={14} /> Download
                </button>
              )}
              <button onClick={loadInsights}
                className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-[var(--muted)] hover:text-[var(--text)] hover:bg-gray-100 rounded-lg transition">
                <RefreshCw size={14} /> Refresh
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-5">
            {insightsLoading ? (
              <div className="flex items-center justify-center h-full">
                <Loader2 size={32} className="animate-spin text-[var(--primary)]" />
              </div>
            ) : insights ? (
              <div className="prose">
                <ReactMarkdown>{insights}</ReactMarkdown>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <p className="text-[var(--muted)] mb-2">⏳ Your campaign insights are still being generated.</p>
                <p className="text-sm text-[var(--muted)]">This typically takes 1-2 minutes.</p>
                <button onClick={loadInsights}
                  className="mt-4 px-4 py-2 bg-[var(--primary)] text-white rounded-lg text-sm font-medium hover:bg-[var(--primary-dark)] transition">
                  Check Again
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Chat panel */}
        <div className="w-full md:w-1/2 flex flex-col bg-[#fafafa]">
          <div className="px-5 py-3 border-b border-[var(--border)] bg-white flex items-center justify-between shrink-0">
            <h2 className="font-semibold flex items-center gap-2"><MessageSquare size={18} /> Chat</h2>
            <button onClick={() => setMessages([])}
              className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-[var(--danger)] hover:bg-red-50 rounded-lg transition">
              <Trash2 size={14} /> Clear
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-5 space-y-4">
            {messages.length === 0 && (
              <div className="flex items-center justify-center h-full text-[var(--muted)] text-sm">
                Start a conversation about your campaign strategy.
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed
                  ${msg.role === "user"
                    ? "bg-[var(--primary)] text-white rounded-br-md"
                    : "bg-white border border-[var(--border)] rounded-bl-md"}`}>
                  {msg.role === "assistant" ? (
                    <div className="prose text-sm">
                      <ReactMarkdown>{msg.content[0].text}</ReactMarkdown>
                    </div>
                  ) : msg.content[0].text}
                </div>
              </div>
            ))}
            {sending && (
              <div className="flex justify-start">
                <div className="bg-white border border-[var(--border)] rounded-2xl rounded-bl-md px-4 py-3">
                  <Loader2 size={18} className="animate-spin text-[var(--primary)]" />
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 bg-white border-t border-[var(--border)] shrink-0">
            <div className="flex gap-2">
              <textarea value={input} onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
                placeholder="e.g., How should I focus my door-knocking efforts?"
                rows={2}
                className="flex-1 px-4 py-2.5 border border-[var(--border)] rounded-xl resize-none focus:ring-2 focus:ring-[var(--primary)] outline-none text-sm" />
              <button onClick={sendMessage} disabled={sending || !input.trim()}
                className="px-4 bg-[var(--accent)] hover:bg-green-600 text-white rounded-xl transition disabled:opacity-40 self-end">
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
