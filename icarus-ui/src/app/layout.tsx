import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/lib/auth-context";

export const metadata: Metadata = {
  title: "WinFlip",
  description: "Campaign Strategy AI",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[var(--bg)]">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
