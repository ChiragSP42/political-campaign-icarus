"use client";

import { useEffect, useCallback } from "react";
import { Plus, Trash2, MessageSquare } from "lucide-react";

export interface ChatSession {
  chatId: string;
  title: string;
  createdAt: string;
}

interface ChatSidebarProps {
  email: string;
  activeChatId: string | null;
  onSelectSession: (chatId: string) => void;
  onNewChat: () => void;
  sessions: ChatSession[];
  setSessions: React.Dispatch<React.SetStateAction<ChatSession[]>>;
}

function timeAgo(dateString: string): string {
  const now = Date.now();
  const then = new Date(dateString).getTime();
  const seconds = Math.floor((now - then) / 1000);

  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

export default function ChatSidebar({
  email,
  activeChatId,
  onSelectSession,
  onNewChat,
  sessions,
  setSessions,
}: ChatSidebarProps) {
  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch(
        `/api/chat/sessions?email=${encodeURIComponent(email)}`
      );
      const data = await res.json();
      if (Array.isArray(data)) {
        setSessions(data);
      }
    } catch {
      // keep sidebar empty on error
    }
  }, [email, setSessions]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  async function handleDelete(chatId: string) {
    try {
      const res = await fetch(
        `/api/chat/sessions?chatId=${encodeURIComponent(chatId)}&email=${encodeURIComponent(email)}`,
        { method: "DELETE" }
      );
      if (res.ok) {
        setSessions((prev) => prev.filter((s) => s.chatId !== chatId));
      }
    } catch {
      // keep session in list on error
    }
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        borderRight: "1px solid var(--border)",
        background: "var(--card)",
      }}
    >
      {/* New Chat button */}
      <div style={{ padding: "12px", borderBottom: "1px solid var(--border)" }}>
        <button
          onClick={onNewChat}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            width: "100%",
            padding: "8px 12px",
            background: "var(--primary)",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            fontSize: "13px",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          <Plus size={16} /> New Chat
        </button>
      </div>

      {/* Session list */}
      <div style={{ flex: 1, overflowY: "auto", padding: "8px" }}>
        {sessions.length === 0 && (
          <p
            style={{
              textAlign: "center",
              color: "var(--muted)",
              fontSize: "13px",
              marginTop: "24px",
            }}
          >
            No conversations yet
          </p>
        )}
        {sessions.map((session) => {
          const isActive = session.chatId === activeChatId;
          return (
            <div
              key={session.chatId}
              onClick={() => onSelectSession(session.chatId)}
              style={{
                display: "flex",
                alignItems: "flex-start",
                gap: "8px",
                padding: "10px",
                marginBottom: "4px",
                borderRadius: "8px",
                cursor: "pointer",
                background: isActive ? "#eef2ff" : "transparent",
                border: isActive
                  ? "1px solid var(--primary)"
                  : "1px solid transparent",
                transition: "background 0.15s",
              }}
              onMouseEnter={(e) => {
                if (!isActive)
                  (e.currentTarget as HTMLDivElement).style.background =
                    "#f8fafc";
              }}
              onMouseLeave={(e) => {
                if (!isActive)
                  (e.currentTarget as HTMLDivElement).style.background =
                    "transparent";
              }}
            >
              <MessageSquare
                size={16}
                style={{
                  marginTop: "2px",
                  flexShrink: 0,
                  color: isActive ? "var(--primary)" : "var(--muted)",
                }}
              />
              <div style={{ flex: 1, minWidth: 0 }}>
                <p
                  style={{
                    fontSize: "13px",
                    fontWeight: 500,
                    color: "var(--text)",
                    margin: 0,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {session.title}
                </p>
                <p
                  style={{
                    fontSize: "11px",
                    color: "var(--muted)",
                    margin: "2px 0 0",
                  }}
                >
                  {timeAgo(session.createdAt)}
                </p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(session.chatId);
                }}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  padding: "4px",
                  borderRadius: "4px",
                  color: "var(--muted)",
                  flexShrink: 0,
                  display: "flex",
                  alignItems: "center",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.color =
                    "var(--danger)";
                  (e.currentTarget as HTMLButtonElement).style.background =
                    "#fef2f2";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.color =
                    "var(--muted)";
                  (e.currentTarget as HTMLButtonElement).style.background =
                    "none";
                }}
                title="Delete session"
              >
                <Trash2 size={14} />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
