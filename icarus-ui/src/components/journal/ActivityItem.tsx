import React from "react";

export interface ActivityItemProps {
  title: string;
  tags: { region?: string; topic?: string };
}

export function ActivityItem({ title, tags }: ActivityItemProps) {
  const tagParts = [tags.region, tags.topic].filter(Boolean);

  return (
    <div className="py-2">
      <p className="font-bold text-[var(--text)]">{title}</p>
      {tagParts.length > 0 && (
        <p className="text-sm text-[var(--muted)]">{tagParts.join(" • ")}</p>
      )}
    </div>
  );
}
