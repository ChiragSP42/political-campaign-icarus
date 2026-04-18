export interface Priority {
  id: string;
  label: string;
  text: string;
}

export interface ActivityEntry {
  id: string;
  title: string;
  region?: string;
  topic?: string;
  createdAt: string;
}

export interface Insight {
  id: string;
  text: string;
  confirmed: boolean;
}

export interface SuggestedAction {
  id: string;
  text: string;
  done: boolean;
}

export type FilterCategory = "Location" | "Topic" | "Event" | "Date";
