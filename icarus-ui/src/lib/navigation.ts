export interface NavItem {
  label: string;
  href: string;
  /** Paths that should mark this item as active (prefix match) */
  activePaths: string[];
}

export const PRIMARY_NAV_ITEMS: NavItem[] = [
  { label: "Dashboard", href: "/dashboard", activePaths: ["/dashboard"] },
  { label: "Questionnaire", href: "/questionnaire", activePaths: ["/questionnaire"] },
  { label: "Journal", href: "/journal", activePaths: ["/journal"] },
];

export const JOURNAL_SUB_NAV_ITEMS: NavItem[] = [
  { label: "Overview", href: "/journal/overview", activePaths: ["/journal/overview", "/journal"] },
  { label: "Entries", href: "/journal/entries", activePaths: ["/journal/entries"] },
  { label: "Insights", href: "/journal/insights", activePaths: ["/journal/insights"] },
  { label: "Actions", href: "/journal/actions", activePaths: ["/journal/actions"] },
];

export function isActive(currentPath: string, activePaths: string[]): boolean {
  return activePaths.some((p) => currentPath === p || currentPath.startsWith(p + "/"));
}
