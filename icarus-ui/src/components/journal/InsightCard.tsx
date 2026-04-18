import React from "react";
import { Card } from "@/components/shared";
import { Button } from "@/components/shared";

export interface InsightCardProps {
  text: string;
  onConfirm: () => void;
  onEdit: () => void;
}

export function InsightCard({ text, onConfirm, onEdit }: InsightCardProps) {
  return (
    <Card>
      <p className="text-[var(--text)] mb-3">{text}</p>
      <div className="flex gap-2">
        <Button variant="primary" size="sm" onClick={onConfirm}>
          Confirm
        </Button>
        <Button variant="secondary" size="sm" onClick={onEdit}>
          Edit
        </Button>
      </div>
    </Card>
  );
}
