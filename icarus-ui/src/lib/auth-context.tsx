"use client";

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";
import { useRouter } from "next/navigation";

interface AuthState {
  email: string | null;
  authenticated: boolean;
  questionnaireCompleted: boolean;
  loading: boolean;
}

interface AuthContextType extends AuthState {
  setUser: (email: string, questionnaireCompleted: boolean) => void;
  logout: () => void;
  refreshSession: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuth] = useState<AuthState>({
    email: null,
    authenticated: false,
    questionnaireCompleted: false,
    loading: true,
  });

  const checkSession = useCallback(async () => {
    try {
      const res = await fetch("/api/auth/session");
      if (res.ok) {
        const data = await res.json();
        setAuth({
          email: data.email,
          authenticated: true,
          questionnaireCompleted: data.questionnaireCompleted,
          loading: false,
        });
      } else {
        setAuth({ email: null, authenticated: false, questionnaireCompleted: false, loading: false });
      }
    } catch {
      setAuth({ email: null, authenticated: false, questionnaireCompleted: false, loading: false });
    }
  }, []);

  useEffect(() => {
    checkSession();
  }, [checkSession]);

  const setUser = (email: string, questionnaireCompleted: boolean) => {
    setAuth({ email, authenticated: true, questionnaireCompleted, loading: false });
  };

  const logout = async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } catch { /* best effort */ }
    setAuth({ email: null, authenticated: false, questionnaireCompleted: false, loading: false });
  };

  const refreshSession = async () => {
    await checkSession();
  };

  return (
    <AuthContext.Provider value={{ ...auth, setUser, logout, refreshSession }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}

export function useRequireAuth() {
  const auth = useAuth();
  const router = useRouter();
  useEffect(() => {
    if (!auth.loading && !auth.authenticated) router.replace("/auth");
  }, [auth.loading, auth.authenticated, router]);
  return auth;
}
