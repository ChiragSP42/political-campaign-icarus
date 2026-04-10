# Requirements Document

## Introduction

This feature reworks the WinFlip/icarus-ui Next.js frontend from its current monolithic page-based structure into a modular component architecture with a unified top navigation bar. The current app has all UI logic embedded directly in page files (`auth/page.tsx`, `dashboard/page.tsx`, `questionnaire/page.tsx`) with no shared component library. This rework introduces a top navigation bar with hamburger menu (replacing any sidebar patterns), a modular component system for reusability, and a new Campaign Journal section derived from UI mockups. All existing pages (auth, questionnaire, dashboard) will adopt the new layout.

## Glossary

- **Top_Nav**: The persistent horizontal navigation bar rendered at the top of every authenticated page, containing the app logo/title, primary section links, user controls, and the Hamburger_Menu trigger
- **Hamburger_Menu**: A slide-out or dropdown panel triggered by a three-line icon in the Top_Nav, revealing secondary/contextual navigation items (e.g., Journal sub-pages: Overview, Entries, Insights, Actions)
- **App_Shell**: The root layout component that wraps all authenticated pages, composed of the Top_Nav and a content area below it
- **Journal_Module**: The Campaign Journal feature area containing the Overview, Entries, Insights, and Actions sub-pages
- **Filter_Chip**: A pill-shaped interactive button used to filter content by category (Location, Topic, Event, Date)
- **Priority_Card**: A card component displaying a priority label and descriptive text, rendered in a horizontal row on the Journal Overview
- **Activity_Item**: A list item component showing a title and metadata tags (e.g., region and topic) in the Recent Activity section
- **Insight_Card**: A card component displaying insight text with Confirm and Edit action buttons
- **Action_Item**: A list item component displaying a suggested action with a Done button
- **Component_Registry**: The `src/components/` directory structure organized by domain (layout, journal, shared) enabling independent component development
- **Auth_Page**: The sign-in/sign-up page at `/auth`, which renders without the Top_Nav since the user is not yet authenticated
- **Dashboard_Page**: The existing chat-based campaign strategist page at `/dashboard`
- **Questionnaire_Page**: The candidate intake questionnaire page at `/questionnaire`

## Requirements

### Requirement 1: Top Navigation Bar

**User Story:** As an authenticated user, I want a persistent top navigation bar on every authenticated page, so that I can navigate between major app sections without a sidebar taking up horizontal space.

#### Acceptance Criteria

1. WHEN an authenticated user loads any page other than Auth_Page, THE App_Shell SHALL render the Top_Nav as a fixed horizontal bar at the top of the viewport.
2. THE Top_Nav SHALL display the application logo or title on the left side, primary navigation links (Dashboard, Questionnaire, Journal) in the center or right area, and user controls (email display, logout button) on the far right.
3. THE Top_Nav SHALL visually indicate the currently active section by applying a distinct style (e.g., underline, bold, or highlight) to the corresponding navigation link.
4. WHEN the user clicks a primary navigation link in the Top_Nav, THE App_Shell SHALL navigate to the corresponding route (`/dashboard`, `/questionnaire`, `/journal`).
5. THE Top_Nav SHALL remain visible and fixed at the top of the viewport during page scrolling.
6. WHEN the viewport width is below 768px, THE Top_Nav SHALL collapse the primary navigation links and display only the Hamburger_Menu trigger and the logo.

### Requirement 2: Hamburger Menu

**User Story:** As a user, I want a hamburger menu that reveals deeper navigation options, so that I can access sub-pages and secondary navigation without cluttering the top bar.

#### Acceptance Criteria

1. WHEN the user clicks the Hamburger_Menu trigger icon (three horizontal lines) in the Top_Nav, THE Hamburger_Menu SHALL open a panel displaying contextual navigation items.
2. THE Hamburger_Menu panel SHALL display all primary section links (Dashboard, Questionnaire, Journal) and, when the user is within the Journal_Module, SHALL additionally display Journal sub-page links (Overview, Entries, Insights, Actions).
3. WHEN the user clicks a navigation item within the Hamburger_Menu, THE App_Shell SHALL navigate to the selected route and close the Hamburger_Menu.
4. WHEN the user clicks outside the Hamburger_Menu panel or presses the Escape key, THE Hamburger_Menu SHALL close.
5. WHILE the Hamburger_Menu is open, THE Hamburger_Menu SHALL render a semi-transparent backdrop overlay behind the panel to indicate modal context.
6. THE Hamburger_Menu trigger icon SHALL animate to an X (close) icon while the menu is open.

### Requirement 3: Modular Component Architecture

**User Story:** As a developer, I want UI components organized in a modular directory structure, so that I can add, remove, and modify components independently without affecting unrelated parts of the application.

#### Acceptance Criteria

1. THE Component_Registry SHALL organize components into domain directories: `src/components/layout/` for layout components (Top_Nav, App_Shell, Hamburger_Menu), `src/components/journal/` for Journal_Module components, and `src/components/shared/` for reusable primitives (Filter_Chip, cards, buttons).
2. THE Component_Registry SHALL export each component as a named export from its own file, with one component per file.
3. THE Component_Registry SHALL provide barrel export files (`index.ts`) in each domain directory to enable clean import paths.
4. WHEN a new component is added to a domain directory, THE Component_Registry SHALL require only adding the component file and updating the barrel export, with no changes to unrelated components.
5. THE App_Shell SHALL compose the Top_Nav and page content area as independent child components, enabling replacement or modification of either without affecting the other.
6. WHEN the Auth_Page is rendered, THE App_Shell SHALL render only the page content without the Top_Nav, since the user is not authenticated.

### Requirement 4: Journal Overview Page

**User Story:** As a campaign manager, I want a Journal Overview page that shows my priorities, recent activity, new insights, and suggested actions at a glance, so that I can quickly assess campaign status and take action.

#### Acceptance Criteria

1. WHEN an authenticated user navigates to `/journal` or `/journal/overview`, THE Journal_Module SHALL render the Overview page as the default view.
2. THE Overview page SHALL display a heading "Overview" and a "+ New Entry" button in the page header area.
3. THE Overview page SHALL render a row of Filter_Chip components for the categories: Location, Topic, Event, and Date.
4. THE Overview page SHALL render a "Priorities" section displaying up to three Priority_Card components in a horizontal row, each showing a "Priority" label and descriptive text.
5. THE Overview page SHALL render a "Recent Activity" section displaying a vertical list of Activity_Item components, each showing a title and metadata tags (region and topic).
6. THE Overview page SHALL render a "New Insights" section displaying Insight_Card components, each showing insight text with a "Confirm" button (dark/navy style) and an "Edit" button (light/outline style).
7. THE Overview page SHALL render a "Suggested Actions" section displaying Action_Item components, each showing action text with a "Done" button (dark/navy style).

### Requirement 5: Journal Sub-Page Routing

**User Story:** As a user, I want to navigate between Journal sub-pages (Overview, Entries, Insights, Actions), so that I can access different aspects of my campaign journal.

#### Acceptance Criteria

1. THE Journal_Module SHALL define routes for four sub-pages: `/journal/overview` (or `/journal`), `/journal/entries`, `/journal/insights`, and `/journal/actions`.
2. WHEN the user is within the Journal_Module, THE Top_Nav or Hamburger_Menu SHALL display sub-navigation links for Overview, Entries, Insights, and Actions.
3. WHEN the user clicks a Journal sub-navigation link, THE Journal_Module SHALL navigate to the corresponding sub-page route.
4. THE Journal sub-navigation SHALL visually indicate the currently active sub-page.
5. WHEN the user navigates to `/journal` without a sub-path, THE Journal_Module SHALL redirect to `/journal/overview` or render the Overview page as the default.

### Requirement 6: Existing Page Migration to App Shell

**User Story:** As a user, I want the existing Dashboard and Questionnaire pages to use the same top navigation layout as the new Journal section, so that the app has a consistent look and navigation experience.

#### Acceptance Criteria

1. WHEN the Dashboard_Page is rendered, THE App_Shell SHALL wrap the dashboard content with the Top_Nav, and the Dashboard_Page SHALL retain all existing functionality (insights panel, chat panel, header controls).
2. WHEN the Questionnaire_Page is rendered, THE App_Shell SHALL wrap the questionnaire content with the Top_Nav, and the Questionnaire_Page SHALL retain all existing functionality (step navigation, form fields, submission, polling).
3. THE Dashboard_Page SHALL remove its existing inline header and rely on the Top_Nav for navigation and user controls (email display, logout, edit questionnaire link).
4. WHEN the Auth_Page is rendered, THE App_Shell SHALL render the Auth_Page content without the Top_Nav, preserving the current centered card layout.
5. THE root page (`/`) redirect logic SHALL continue to route unauthenticated users to Auth_Page, users without completed questionnaires to Questionnaire_Page, and fully onboarded users to Dashboard_Page.

### Requirement 7: Shared UI Primitives

**User Story:** As a developer, I want reusable shared UI primitives (buttons, cards, chips), so that I can build new features with consistent styling and reduce code duplication across pages.

#### Acceptance Criteria

1. THE Component_Registry SHALL provide a reusable Button component in `src/components/shared/` that supports variant props for primary (dark/navy), secondary (light/outline), and danger styles.
2. THE Component_Registry SHALL provide a reusable Card component in `src/components/shared/` that renders a bordered, rounded container with configurable padding and optional header.
3. THE Component_Registry SHALL provide a reusable Filter_Chip component in `src/components/shared/` that renders a pill-shaped toggle button with active and inactive visual states.
4. WHEN a shared primitive is used across multiple pages, THE shared primitive SHALL accept props for customization (e.g., onClick, label, variant, className) without requiring page-specific logic inside the primitive.
5. THE shared primitives SHALL use the existing CSS custom properties (`--primary`, `--accent`, `--border`, `--text`, `--muted`) defined in `globals.css` for consistent theming.

### Requirement 8: Responsive Layout

**User Story:** As a user on a mobile device, I want the app layout to adapt to smaller screens, so that I can use the Campaign Journal and other pages comfortably on any device.

#### Acceptance Criteria

1. WHEN the viewport width is below 768px, THE Top_Nav SHALL collapse primary navigation links into the Hamburger_Menu and display only the logo and hamburger icon.
2. WHEN the viewport width is below 768px, THE Priority_Card components on the Journal Overview SHALL stack vertically instead of displaying in a horizontal row.
3. WHEN the viewport width is below 768px, THE Dashboard_Page insights panel and chat panel SHALL stack vertically instead of displaying side by side.
4. THE App_Shell content area SHALL use a maximum width constraint on large screens and center the content horizontally, with appropriate padding on all screen sizes.
5. WHILE the viewport width is at or above 768px, THE Top_Nav SHALL display all primary navigation links inline without requiring the Hamburger_Menu for primary navigation.
