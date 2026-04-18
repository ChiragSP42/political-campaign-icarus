"use client";

import React, { useState } from "react";
import { Button, FilterChip } from "@/components/shared";
import { PriorityCard } from "./PriorityCard";
import { ActivityItem } from "./ActivityItem";
import { InsightCard } from "./InsightCard";
import { ActionItem } from "./ActionItem";
import type { FilterCategory } from "@/lib/journal-types";

const FILTER_CATEGORIES: FilterCategory[] = ["Location", "Topic", "Event", "Date"];

const PRIORITIES = [
  { id: "1", label: "Priority", text: "Focus on housing messaging" },
  { id: "2", label: "Priority", text: "Focus on housing messaging" },
  { id: "3", label: "Priority", text: "Focus on housing messaging" },
];

const ACTIVITIES = [
  { id: "1", title: "Town hall with voters", tags: { region: "Montco", topic: "Education" } },
  { id: "2", title: "Chat about education strategy", tags: { region: "Montco", topic: "Education" } },
  { id: "3", title: "Fundraising meeting", tags: { region: "Montco", topic: "Education" } },
];

const INSIGHTS = [
  { id: "1", text: "Voters are more concerned about affordability than education" },
];

const ACTIONS = [
  { id: "1", text: "Adjust messaging to affordability" },
  { id: "2", text: "Attend housing forum" },
];

export function OverviewPage() {
  const [activeFilters, setActiveFilters] = useState<Set<FilterCategory>>(new Set());

  function toggleFilter(category: FilterCategory) {
    setActiveFilters((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-[var(--text)]">Overview</h1>
        <Button variant="primary" size="md">
          + New Entry
        </Button>
      </div>

      {/* Filter Chips */}
      <div className="flex flex-wrap gap-2">
        {FILTER_CATEGORIES.map((category) => (
          <FilterChip
            key={category}
            label={category}
            active={activeFilters.has(category)}
            onClick={() => toggleFilter(category)}
          />
        ))}
      </div>

      {/* Priorities */}
      <section>
        <h2 className="text-lg font-semibold text-[var(--text)] mb-3">Priorities</h2>
        <div className="flex flex-col md:flex-row gap-4">
          {PRIORITIES.map((p) => (
            <div key={p.id} className="flex-1">
              <PriorityCard label={p.label} text={p.text} />
            </div>
          ))}
        </div>
      </section>

      {/* Recent Activity */}
      <section>
        <h2 className="text-lg font-semibold text-[var(--text)] mb-3">Recent Activity</h2>
        <div className="divide-y divide-[var(--border)]">
          {ACTIVITIES.map((a) => (
            <ActivityItem key={a.id} title={a.title} tags={a.tags} />
          ))}
        </div>
      </section>

      {/* New Insights */}
      <section>
        <h2 className="text-lg font-semibold text-[var(--text)] mb-3">New Insights</h2>
        <div className="space-y-3">
          {INSIGHTS.map((i) => (
            <InsightCard
              key={i.id}
              text={i.text}
              onConfirm={() => {}}
              onEdit={() => {}}
            />
          ))}
        </div>
      </section>

      {/* Suggested Actions */}
      <section>
        <h2 className="text-lg font-semibold text-[var(--text)] mb-3">Suggested Actions</h2>
        <div className="divide-y divide-[var(--border)]">
          {ACTIONS.map((a) => (
            <ActionItem key={a.id} text={a.text} onDone={() => {}} />
          ))}
        </div>
      </section>
    </div>
  );
}
