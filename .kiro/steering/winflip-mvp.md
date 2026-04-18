---
inclusion: auto
---

# WinFlip MVP — Project Steering

## Project Overview

WinFlip is an AI-powered campaign copiloting platform for down-ballot Virginia candidates. The project evolved from a Phase 1 POC ("Project Icarus" — Streamlit + Lambda + Bedrock + S3) into the current MVP with a Next.js frontend, DynamoDB storage, campaign journal, and enhanced AI features. The codebase still uses the `icarus-` naming in folder structure from Phase 1 but is being branded as WinFlip.

## Code Location

- Frontend: `icarus-ui/` (Next.js App Router)
- Backend lambdas: `icarus-cdk/services/lambdas/`
- CDK infrastructure: `icarus-cdk/infra/`
- Prompt templates & local data: `icarus-cdk/local-files/`
- Shared Python helpers: `aws_helpers/`
- Phase 1 prototype reference: `prototype_chatbot_backend/`
- Spec files: `.kiro/specs/`

## Package Management

- Use `npm` for the frontend (`icarus-ui/`).
- Python tooling uses `uv` at the repo root (see `pyproject.toml` and `uv.lock`).
- Lambda functions currently share a single code asset deployed from `icarus-cdk/services/lambdas/`.

## Tech Stack

### Frontend
- Next.js 15 (App Router) with TypeScript
- Tailwind CSS v4 for styling
- AWS Amplify for hosting and CI/CD (`amplify.yml` at repo root)
- AWS SDK v3 called directly from Next.js API routes (no Amplify Auth library)
- React Context (`AuthProvider` in `src/lib/auth-context.tsx`) for auth state, persisted to localStorage
- lucide-react for icons
- react-markdown for rendering markdown

### Backend
- Python 3.13 for all Lambda functions
- boto3 for AWS service interactions
- Amazon Bedrock (Claude Sonnet) for AI operations
- All lambdas deployed from a single asset path (`icarus-cdk/services/lambdas/`)

### Infrastructure (CDK — TypeScript)
- Single stack: `IcarusDannerInfraStack` in `icarus-cdk/infra/lib/infra-stack.ts`
- Amazon Cognito for authentication (email-based sign-up, user-password auth flow)
- API Gateway (REST) with CORS and dev stage
- DynamoDB tables:
  - `chat-history-{accountId}` — partition key `chatId`, sort key `timestamp`, GSI on `userId`/`createdAt`
  - `main-{accountId}` — partition key `userId` (prefixed `USER#`), sort key `SK` (index-overloaded: `META#`, `CHATS#`, `NOTES#`, `TASKS#`, etc.)
- S3 buckets for election data, prompts, questionnaires, generated insights, chatbot responses
- Shared IAM role (`lambda-role`) with Bedrock, S3, and DynamoDB permissions
- Prompt templates auto-deployed from `icarus-cdk/local-files/` to the prompt bucket

## Frontend Structure

```
icarus-ui/src/
├── app/
│   ├── (authenticated)/          # Route group — requires auth
│   │   ├── layout.tsx            # Auth guard + AppShell wrapper
│   │   ├── dashboard/page.tsx
│   │   ├── questionnaire/page.tsx
│   │   └── journal/
│   │       ├── layout.tsx        # Journal sub-navigation
│   │       ├── page.tsx          # Redirects to overview
│   │       ├── overview/page.tsx
│   │       ├── entries/page.tsx
│   │       ├── insights/page.tsx
│   │       └── actions/page.tsx
│   ├── api/
│   │   ├── auth/
│   │   │   ├── signup/route.ts   # Cognito SignUp
│   │   │   ├── confirm/route.ts  # Cognito ConfirmSignUp + DDB user row creation
│   │   │   └── signin/route.ts   # Cognito InitiateAuth + questionnaire check
│   │   ├── chat/
│   │   │   ├── send/route.ts     # Trigger chatbot lambda
│   │   │   ├── check/route.ts    # Poll for chatbot response
│   │   │   └── sessions/         # Session CRUD + messages
│   │   ├── insights/route.ts
│   │   └── questionnaire/save/route.ts
│   ├── auth/page.tsx             # Sign-in / sign-up / verify UI
│   ├── layout.tsx                # Root layout with AuthProvider
│   └── globals.css
├── components/
│   ├── chat/ChatSidebar.tsx
│   ├── journal/                  # ActionItem, InsightCard, OverviewPage, etc.
│   ├── layout/                   # AppShell, TopNav, NavLinks, HamburgerMenu
│   └── shared/                   # Button, Card, FilterChip
└── lib/
    ├── auth-context.tsx          # AuthProvider + useAuth + useRequireAuth
    ├── constants.ts              # Questionnaire options, archetype questions
    ├── journal-types.ts          # Priority, ActivityEntry, Insight, SuggestedAction types
    └── navigation.ts             # Nav items config + isActive helper
```

## Lambda Functions

| File | Purpose |
|------|---------|
| `chatbot_lambda.py` | Main Bedrock-powered chatbot |
| `trigger_chatbot.py` | Async invocation of chatbot lambda, writes pending record to chat history |
| `check_LLM_response_lambda.py` | Polls S3/DDB for chatbot response |
| `check_questionnaire_lambda.py` | Checks if user completed questionnaire (S3 lookup) |
| `save_questionnaire_lambda.py` | Stores questionnaire answers to S3 |
| `generate_insights_lambda.py` | Triggered by S3 PUT on questionnaire bucket, generates insights via Bedrock |
| `session_manager_lambda.py` | Chat session CRUD (list sessions, get messages, delete) |

## DynamoDB Patterns

### `main` table (single-table design)
- PK: `userId` — always prefixed with `USER#` (e.g., `USER#user@example.com`)
- SK: Index-overloaded with entity prefixes:
  - `META#PROFILE` — user profile metadata (email, createdAt, status)
  - `CHATS#<chatId>` — chat session references
  - `NOTES#<noteId>` — user notes
  - `TASKS#<taskId>` — user tasks
- User row is created automatically on email confirmation (`/api/auth/confirm`)
- Uses `ConditionExpression: attribute_not_exists(userId)` to prevent duplicates

### `chat-history` table
- PK: `chatId`, SK: `timestamp`
- GSI: `userId-index` (PK: `userId`, SK: `createdAt`) for listing a user's sessions

## Auth Flow

1. **Sign Up** → `POST /api/auth/signup` → Cognito `SignUp` (user unconfirmed)
2. **Verify Email** → `POST /api/auth/confirm` → Cognito `ConfirmSignUp` + creates `USER#<email>` / `META#PROFILE` row in `main` DDB table
3. **Sign In** → `POST /api/auth/signin` → Cognito `InitiateAuth` (USER_PASSWORD_AUTH) + checks questionnaire completion → redirects to `/questionnaire` or `/dashboard`
4. Auth state stored in localStorage via `AuthProvider` context
5. Authenticated routes protected by `useRequireAuth()` hook in the `(authenticated)` layout

## Key Conventions

### API Routes (Next.js)
- All AWS SDK calls happen in Next.js API routes (server-side), not in client components
- Cognito and DynamoDB clients use explicit credentials from env vars (`CUSTOM_ACCESS_KEY_ID`, `CUSTOM_SECRET_ACCESS_KEY`)
- API Gateway endpoints proxied via `API_ENDPOINT` env var for lambda-backed operations

### Frontend
- Components organized by feature: `chat/`, `journal/`, `layout/`, `shared/`
- Type definitions in `src/lib/journal-types.ts`
- Navigation config in `src/lib/navigation.ts`
- Questionnaire constants in `src/lib/constants.ts`
- CSS variables for theming (`--primary`, `--muted`, `--border`, `--bg`, etc.)

### CDK / Infrastructure
- Single stack, not modular constructs (all resources in `infra-stack.ts`)
- S3 bucket naming: `{purpose}-{accountId}` (e.g., `icarus-questionnaires-{accountId}`)
- DynamoDB table naming: `{name}-{accountId}`
- Lambda naming: `{function}-lambda`
- All lambdas share one IAM role with broad S3 + Bedrock + DDB permissions
- Environment variables passed to lambdas for all configurable values

## Environment Variables (Frontend — `.env.local`)

| Variable | Purpose |
|----------|---------|
| `CUSTOM_REGION` | AWS region |
| `CUSTOM_ACCESS_KEY_ID` | AWS access key for SDK calls |
| `CUSTOM_SECRET_ACCESS_KEY` | AWS secret key for SDK calls |
| `COGNITO_USER_POOL_ID` | Cognito User Pool ID |
| `COGNITO_CLIENT_ID` | Cognito App Client ID |
| `COGNITO_REGION` | Cognito region |
| `API_ENDPOINT` | API Gateway base URL |
| `MAIN_TABLE_NAME` | DynamoDB main table name |

## Deployment

- Frontend deployed via AWS Amplify (see `amplify.yml`)
- Amplify build injects `CUSTOM_*`, `COGNITO_*`, and `API_ENDPOINT` env vars into `.env.production`
- CDK infrastructure deployed separately from `icarus-cdk/infra/`

## Out of Scope (MVP)

- Stripe payments / subscription management
- Multi-state data beyond Virginia
- Native mobile apps
- Real-time social media monitoring
- Full NGP VAN / L2 integration
- Production HA/DR
