"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useAuth } from "@/lib/auth-context";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Send, Trash2, Download, RefreshCw, FileText, MessageSquare, Loader2, PanelLeftClose, PanelLeftOpen,
} from "lucide-react";
import ChatSidebar, { ChatSession } from "@/components/chat/ChatSidebar";

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
  const [chatId, setChatId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [leftPanelWidth, setLeftPanelWidth] = useState(50);
  const isDragging = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const loadInsights = useCallback(async () => {
    if (!auth.email) return;
    setInsightsLoading(true);
    try {
      const res = await fetch(`/api/insights?email=${encodeURIComponent(auth.email)}`);
      const data = await res.json();
      if (data.exists) {
        setInsights(data.content);
        return true;
      } else {
        setInsights(null);
        return false;
      }
    } catch {
      setInsights(null);
      return false;
    } finally {
      setInsightsLoading(false);
    }
  }, [auth.email]);

  // Auto-poll for insights when they haven't been generated yet
  useEffect(() => {
    loadInsights();
  }, [loadInsights]);

  useEffect(() => {
    if (insights !== null || !auth.email) return;
    // insights are null — start polling every 10s until they arrive
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/insights?email=${encodeURIComponent(auth.email!)}`);
        const data = await res.json();
        if (data.exists && data.content) {
          setInsights(data.content);
          setInsightsLoading(false);
        }
      } catch { /* keep polling */ }
    }, 10000);
    return () => clearInterval(interval);
  }, [insights, auth.email]);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const pct = ((e.clientX - rect.left) / rect.width) * 100;
      setLeftPanelWidth(Math.min(80, Math.max(20, pct)));
    };
    const handleMouseUp = () => { isDragging.current = false; document.body.style.cursor = ""; document.body.style.userSelect = ""; };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => { window.removeEventListener("mousemove", handleMouseMove); window.removeEventListener("mouseup", handleMouseUp); };
  }, []);

  async function onSelectSession(selectedChatId: string) {
    try {
      const res = await fetch(`/api/chat/sessions/messages?chatId=${encodeURIComponent(selectedChatId)}`);
      const data = await res.json();
      if (Array.isArray(data)) {
        const converted: ChatMsg[] = data.map((m: { role: "user" | "assistant"; content: string }) => ({
          role: m.role,
          content: [{ text: m.content }],
        }));
        setMessages(converted);
        setChatId(selectedChatId);
      }
    } catch {
      // keep current state on error
    }
  }

  function onNewChat() {
    setMessages([]);
    setChatId(null);
  }

  async function sendMessage() {
    if (!input.trim() || sending) return;
    const userMsg: ChatMsg = { role: "user", content: [{ text: input }] };
    setMessages((m) => [...m, userMsg]);
    const query = input;
    setInput("");
    setSending(true);

    const isNewSession = chatId === null;

    try {
      const history = [...messages, userMsg].map((m) => ({ role: m.role, content: m.content }));
      const res = await fetch("/api/chat/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: auth.email, query, conversation_history: history, chatId }),
      });
      const data = await res.json();

      // Store the returned chatId
      const returnedChatId = data.chatId || chatId;
      if (data.chatId) {
        setChatId(data.chatId);
      }

      if (data.status === "COMPLETED") {
        // Poll for response
        while (true) {
          const checkRes = await fetch(
            `/api/chat/check?email=${encodeURIComponent(auth.email!)}&chatId=${encodeURIComponent(returnedChatId || "")}`
          );
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

      // Refresh sessions list after a new session's first message completes
      if (isNewSession && auth.email) {
        try {
          const sessRes = await fetch(`/api/chat/sessions?email=${encodeURIComponent(auth.email)}`);
          const sessData = await sessRes.json();
          if (Array.isArray(sessData)) {
            setSessions(sessData);
          }
        } catch {
          // ignore refresh failure
        }
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
      <div ref={containerRef} className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Chat half — sidebar + chat panel (LEFT) */}
        <div className="flex flex-row overflow-hidden" style={{ width: `${leftPanelWidth}%` }}>
          {/* Chat Sidebar — collapsible */}
          {sidebarOpen && (
            <div className="hidden md:block border-r border-[var(--border)]" style={{ width: "240px", flexShrink: 0 }}>
              <ChatSidebar
                email={auth.email || ""}
                activeChatId={chatId}
                onSelectSession={onSelectSession}
                onNewChat={onNewChat}
                sessions={sessions}
                setSessions={setSessions}
              />
            </div>
          )}

          {/* Chat panel */}
          <div className="flex-1 flex flex-col bg-[#fafafa] min-w-0">
            <div className="px-5 py-3 border-b border-[var(--border)] bg-white flex items-center justify-between shrink-0">
              <div className="flex items-center gap-2">
                <button onClick={() => setSidebarOpen((o) => !o)}
                  className="hidden md:flex items-center justify-center w-7 h-7 rounded-md text-[var(--muted)] hover:text-[var(--text)] hover:bg-gray-100 transition"
                  title={sidebarOpen ? "Hide sessions" : "Show sessions"}>
                  {sidebarOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
                </button>
                <h2 className="font-semibold flex items-center gap-2"><MessageSquare size={18} /> Chat</h2>
              </div>
              <button onClick={() => { setMessages([]); setChatId(null); }}
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
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content[0].text}</ReactMarkdown>
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

        {/* Draggable divider */}
        <div
          className="hidden md:flex items-center justify-center w-2 cursor-col-resize hover:bg-[var(--primary)]/10 active:bg-[var(--primary)]/20 transition-colors group"
          onMouseDown={() => { isDragging.current = true; document.body.style.cursor = "col-resize"; document.body.style.userSelect = "none"; }}
        >
          <div className="w-0.5 h-8 bg-gray-300 rounded-full group-hover:bg-[var(--primary)] transition-colors" />
        </div>

        {/* Insights panel (RIGHT) */}
        <div className="flex-1 border-t md:border-t-0 flex flex-col bg-white min-w-0">
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
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{insights}</ReactMarkdown>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <Loader2 size={28} className="animate-spin text-[var(--primary)] mb-3" />
                <p className="text-[var(--muted)] mb-2">⏳ Your campaign insights are still being generated.</p>
                <p className="text-sm text-[var(--muted)]">We're checking automatically — they'll appear here as soon as they're ready.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
