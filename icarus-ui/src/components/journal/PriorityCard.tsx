import React from "react";
import { Card } from "@/components/shared";

export interface PriorityCardProps {
  label: string;
  text: string;
}

export function PriorityCard({ label, text }: PriorityCardProps) {
  return (
    <Card>
      <p className="text-xs text-[var(--muted)] mb-1">{label}</p>
      <p className="font-bold text-[var(--text)]">{text}</p>
    </Card>
  );
}
