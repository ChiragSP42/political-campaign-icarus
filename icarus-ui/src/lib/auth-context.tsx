"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { useRouter } from "next/navigation";

interface AuthState {
  email: string | null;
  authenticated: boolean;
  questionnaireCompleted: boolean;
}

interface AuthContextType extends AuthState {
  setUser: (email: string, questionnaireCompleted: boolean) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuth] = useState<AuthState>({
    email: null,
    authenticated: false,
    questionnaireCompleted: false,
  });

  useEffect(() => {
    const stored = localStorage.getItem("icarus_auth");
    if (stored) {
      try { setAuth(JSON.parse(stored)); } catch { /* ignore */ }
    }
  }, []);

  const setUser = (email: string, questionnaireCompleted: boolean) => {
    const state = { email, authenticated: true, questionnaireCompleted };
    setAuth(state);
    localStorage.setItem("icarus_auth", JSON.stringify(state));
  };

  const logout = () => {
    setAuth({ email: null, authenticated: false, questionnaireCompleted: false });
    localStorage.removeItem("icarus_auth");
  };

  return (
    <AuthContext.Provider value={{ ...auth, setUser, logout }}>
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
    if (!auth.authenticated) router.replace("/auth");
  }, [auth.authenticated, router]);
  return auth;
}
