import React from "react";
import { Button } from "@/components/shared";

export interface ActionItemProps {
  text: string;
  onDone: () => void;
}

export function ActionItem({ text, onDone }: ActionItemProps) {
  return (
    <div className="flex items-center justify-between py-2">
      <p className="text-[var(--text)]">{text}</p>
      <Button variant="primary" size="sm" onClick={onDone}>
        Done
      </Button>
    </div>
  );
}
