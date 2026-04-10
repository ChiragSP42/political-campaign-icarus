# Implementation Plan: UI Rework — Modular Navigation

## Overview

Convert the icarus-ui frontend from monolithic page files into a modular component architecture with a persistent top navigation bar, organized component registry, and a new Campaign Journal module. Implementation proceeds bottom-up: data/config layer → shared primitives → layout components → journal components → route restructuring → page migration → responsive polish.

## Tasks

- [x] 1. Create navigation config and journal data types
  - [x] 1.1 Create `src/lib/navigation.ts` with `NavItem` interface, `PRIMARY_NAV_ITEMS`, `JOURNAL_SUB_NAV_ITEMS` arrays, and `isActive()` utility function
    - Define the `NavItem` type with `label`, `href`, and `activePaths` fields
    - Export `PRIMARY_NAV_ITEMS` for Dashboard, Questionnaire, Journal
    - Export `JOURNAL_SUB_NAV_ITEMS` for Overview, Entries, Insights, Actions
    - Implement `isActive(currentPath, activePaths)` returning true iff path matches exactly or is a sub-path (prefix + `/`)
    - _Requirements: 1.3, 5.4, 2.2_

  - [ ]* 1.2 Write property test for `isActive` function (Property 1)
    - **Property 1: Navigation active state correctness**
    - Use `fast-check` to generate random path strings and verify `isActive` returns true iff `currentPath` exactly matches or starts with an `activePaths` entry followed by `/`
    - Test file: `src/lib/__tests__/navigation.test.ts`
    - **Validates: Requirements 1.3, 5.4**

  - [x] 1.3 Create `src/lib/journal-types.ts` with `Priority`, `ActivityEntry`, `Insight`, `SuggestedAction`, and `FilterCategory` types
    - Export all interfaces for use by journal components
    - _Requirements: 4.1, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 2. Build shared UI primitives
  - [x] 2.1 Create `src/components/shared/Button.tsx`
    - Implement `ButtonProps` extending native button attributes with `variant` (`primary`, `secondary`, `danger`) and optional `size` (`sm`, `md`)
    - Use CSS custom properties (`--primary`, `--danger`, `--border`) for theming
    - _Requirements: 7.1, 7.4, 7.5_

  - [x] 2.2 Create `src/components/shared/Card.tsx`
    - Implement `CardProps` with `children`, optional `header`, `padding` (`sm`, `md`, `lg`), and `className`
    - Render bordered, rounded container using `var(--border)`
    - _Requirements: 7.2, 7.4, 7.5_

  - [x] 2.3 Create `src/components/shared/FilterChip.tsx`
    - Implement `FilterChipProps` with `label`, `active`, `onClick`
    - Pill-shaped button with active state (filled `var(--primary)` + white text) and inactive state (border + muted text)
    - _Requirements: 7.3, 7.4, 7.5_

  - [x] 2.4 Create `src/components/shared/index.ts` barrel export
    - Export `Button`, `Card`, `FilterChip` as named exports
    - _Requirements: 3.3_

  - [ ]* 2.5 Write unit tests for shared primitives
    - Test Button renders correct styles for each variant
    - Test Card renders with/without header and different padding
    - Test FilterChip renders active and inactive states
    - Test file: `src/components/shared/__tests__/Button.test.tsx`, `Card.test.tsx`, `FilterChip.test.tsx`
    - _Requirements: 7.1, 7.2, 7.3_

- [x] 3. Build layout components
  - [x] 3.1 Create `src/components/layout/NavLinks.tsx`
    - Implement `NavLinksProps` with `currentPath`, `orientation` (`horizontal` | `vertical`), optional `onNavigate`
    - Render links from `PRIMARY_NAV_ITEMS` using Next.js `Link`
    - Apply active styling (underline + bold) using `isActive()`
    - Call `onNavigate` after click (for hamburger menu close)
    - _Requirements: 1.2, 1.3, 1.4_

  - [x] 3.2 Create `src/components/layout/UserControls.tsx`
    - Read auth context internally (no props)
    - Display `auth.email` (gracefully handle null)
    - Render logout button that calls `auth.logout()` and navigates to `/auth`
    - _Requirements: 1.2, 6.3_

  - [x] 3.3 Create `src/components/layout/HamburgerMenu.tsx`
    - Implement `HamburgerMenuProps` with `isOpen`, `onClose`, `currentPath`
    - Slide-in panel with semi-transparent backdrop overlay
    - Show all primary nav links; additionally show journal sub-links when `currentPath` starts with `/journal`
    - Close on link click, backdrop click, or Escape key press
    - Animate trigger icon between ☰ and ✕
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 3.4 Write property test for hamburger menu contextual links (Property 2)
    - **Property 2: Hamburger menu contextual links**
    - Use `fast-check` to generate random path strings and verify journal sub-links are included iff path starts with `/journal`
    - Extract the link-filtering logic into a testable pure function (e.g., `getHamburgerLinks(currentPath)`)
    - Test file: `src/components/layout/__tests__/HamburgerMenu.test.ts`
    - **Validates: Requirements 2.2, 5.2**

  - [x] 3.5 Create `src/components/layout/TopNav.tsx`
    - Implement `TopNavProps` with `currentPath`
    - Fixed position, full-width, white background, bottom border
    - Left: app logo/title ("🎭 Project Icarus")
    - Center/right (≥768px via Tailwind `md:`): render `<NavLinks orientation="horizontal" />`
    - Far right: render `<UserControls />`
    - <768px: hide NavLinks, show only logo + hamburger trigger
    - Manage hamburger menu open/close state
    - _Requirements: 1.1, 1.2, 1.5, 1.6, 8.1, 8.5_

  - [x] 3.6 Create `src/components/layout/AppShell.tsx`
    - Implement `AppShellProps` with `children`
    - Render `<TopNav />` fixed at top (using `usePathname()`)
    - Render `children` in `<main>` with top padding to clear fixed nav, max-width constraint, horizontal centering
    - _Requirements: 1.1, 3.5, 8.4_

  - [x] 3.7 Create `src/components/layout/index.ts` barrel export
    - Export `AppShell`, `TopNav`, `HamburgerMenu`, `NavLinks`, `UserControls`
    - _Requirements: 3.3_

  - [ ]* 3.8 Write unit tests for layout components
    - Test AppShell renders TopNav and children
    - Test TopNav displays logo, nav links on desktop, user controls
    - Test TopNav hides nav links and shows hamburger at <768px
    - Test HamburgerMenu opens/closes on trigger, backdrop, Escape
    - Test NavLinks highlights correct active link
    - Test file: `src/components/layout/__tests__/AppShell.test.tsx`, `TopNav.test.tsx`, `HamburgerMenu.test.tsx`, `NavLinks.test.tsx`
    - _Requirements: 1.1, 1.2, 1.3, 1.6, 2.1, 2.4, 2.5, 2.6_

- [x] 4. Checkpoint — Shared primitives and layout components
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Build journal components
  - [x] 5.1 Create `src/components/journal/JournalSubNav.tsx`
    - Implement `JournalSubNavProps` with `currentPath`
    - Horizontal tab bar rendering `JOURNAL_SUB_NAV_ITEMS` links
    - Active tab gets underline/highlight styling using `isActive()`
    - _Requirements: 5.2, 5.4_

  - [x] 5.2 Create `src/components/journal/PriorityCard.tsx`
    - Implement `PriorityCardProps` with `label` and `text`
    - Render using shared `Card` component
    - _Requirements: 4.4, 7.2_

  - [x] 5.3 Create `src/components/journal/ActivityItem.tsx`
    - Implement `ActivityItemProps` with `title` and `tags` (`region?`, `topic?`)
    - Render title and metadata tag pills
    - _Requirements: 4.5_

  - [x] 5.4 Create `src/components/journal/InsightCard.tsx`
    - Implement `InsightCardProps` with `text`, `onConfirm`, `onEdit`
    - Render insight text with Confirm (primary) and Edit (secondary) buttons using shared `Button`
    - _Requirements: 4.6, 7.1_

  - [x] 5.5 Create `src/components/journal/ActionItem.tsx`
    - Implement `ActionItemProps` with `text` and `onDone`
    - Render action text with Done button (primary) using shared `Button`
    - _Requirements: 4.7, 7.1_

  - [x] 5.6 Create `src/components/journal/OverviewPage.tsx`
    - Render page header with "Overview" heading and "+ New Entry" button
    - Render FilterChip row for Location, Topic, Event, Date categories
    - Render Priorities section with up to 3 PriorityCard components in horizontal row (stacks vertically <768px)
    - Render Recent Activity section with ActivityItem list
    - Render New Insights section with InsightCard components
    - Render Suggested Actions section with ActionItem components
    - Use placeholder/static data initially
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 8.2_

  - [x] 5.7 Create `src/components/journal/index.ts` barrel export
    - Export `JournalSubNav`, `OverviewPage`, `PriorityCard`, `ActivityItem`, `InsightCard`, `ActionItem`
    - _Requirements: 3.3_

  - [ ]* 5.8 Write unit tests for journal components
    - Test JournalSubNav renders four tabs with correct active state
    - Test OverviewPage renders all four sections and FilterChips
    - Test PriorityCard, ActivityItem, InsightCard, ActionItem render content and handle callbacks
    - Test file: `src/components/journal/__tests__/JournalSubNav.test.tsx`, `OverviewPage.test.tsx`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.2, 5.4_

- [x] 6. Checkpoint — All components built
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Restructure routes with `(authenticated)` group and journal pages
  - [x] 7.1 Create `src/app/(authenticated)/layout.tsx`
    - Import and render `AppShell` wrapping `children`
    - Use `useRequireAuth()` for auth guard
    - Show loading spinner while auth state is initializing
    - _Requirements: 1.1, 3.5, 3.6, 6.4_

  - [x] 7.2 Create `src/app/(authenticated)/journal/layout.tsx`
    - Render `JournalSubNav` (using `usePathname()`) above `children`
    - _Requirements: 5.2, 5.4_

  - [x] 7.3 Create `src/app/(authenticated)/journal/page.tsx`
    - Redirect to `/journal/overview` (using `redirect()` or client-side router)
    - _Requirements: 5.1, 5.5_

  - [x] 7.4 Create `src/app/(authenticated)/journal/overview/page.tsx`
    - Import and render `OverviewPage` component
    - _Requirements: 4.1_

  - [x] 7.5 Create placeholder pages for remaining journal sub-routes
    - `src/app/(authenticated)/journal/entries/page.tsx` — placeholder "Entries" page
    - `src/app/(authenticated)/journal/insights/page.tsx` — placeholder "Insights" page
    - `src/app/(authenticated)/journal/actions/page.tsx` — placeholder "Actions" page
    - _Requirements: 5.1_

- [x] 8. Migrate existing pages into the App Shell
  - [x] 8.1 Move Dashboard page to `src/app/(authenticated)/dashboard/page.tsx`
    - Copy existing dashboard logic into the new route location
    - Remove the inline header (logo, email, logout, edit questionnaire link) since TopNav now provides these
    - Retain all existing functionality (insights panel, chat panel)
    - Make insights and chat panels stack vertically below 768px using Tailwind `md:` classes
    - Delete old `src/app/dashboard/page.tsx`
    - _Requirements: 6.1, 6.3, 8.3_

  - [x] 8.2 Move Questionnaire page to `src/app/(authenticated)/questionnaire/page.tsx`
    - Copy existing questionnaire logic into the new route location
    - Retain all existing functionality (step navigation, form fields, submission, polling)
    - Remove `useRequireAuth()` from the page itself (handled by authenticated layout)
    - Delete old `src/app/questionnaire/page.tsx`
    - _Requirements: 6.2_

  - [x] 8.3 Verify Auth page remains standalone at `src/app/auth/page.tsx`
    - Confirm Auth page is outside the `(authenticated)` group and renders without TopNav
    - No changes needed to auth page content
    - _Requirements: 3.6, 6.4_

  - [x] 8.4 Verify root page (`src/app/page.tsx`) redirect logic still works
    - Confirm unauthenticated → `/auth`, no questionnaire → `/questionnaire`, onboarded → `/dashboard`
    - _Requirements: 6.5_

- [x] 9. Responsive layout adjustments
  - [x] 9.1 Ensure TopNav collapses nav links into hamburger menu below 768px
    - Use Tailwind `hidden md:flex` / `md:hidden` patterns — no JS resize listeners
    - _Requirements: 1.6, 8.1, 8.5_

  - [x] 9.2 Ensure Journal Overview PriorityCards stack vertically below 768px
    - Use Tailwind `flex-col md:flex-row` on the priorities container
    - _Requirements: 8.2_

  - [x] 9.3 Ensure Dashboard panels stack vertically below 768px
    - Change the insights/chat flex container to `flex-col md:flex-row` with full-width children on mobile
    - _Requirements: 8.3_

  - [x] 9.4 Ensure AppShell content area has max-width constraint and centered layout with padding
    - Apply `max-w-7xl mx-auto px-4 md:px-6` or similar to the main content wrapper
    - _Requirements: 8.4_

- [x] 10. Final checkpoint — Full integration
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate the two correctness properties from the design (isActive logic and hamburger menu contextual links)
- Unit tests validate specific rendering and interaction behavior
- All responsive behavior uses Tailwind CSS `md:` breakpoint — no JS resize listeners — to avoid hydration mismatches
- Journal pages use placeholder/static data initially; backend integration is a separate concern
