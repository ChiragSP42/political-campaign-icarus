---
inclusion: auto
---

# WinFlip Phase 2 MVP — Project Steering

## Project Overview

WinFlip is an AI-powered campaign copiloting platform for down-ballot Virginia candidates. Phase 2 MVP evolves the Phase 1 "Project Icarus" POC (Streamlit + Lambda + Bedrock + S3) into a production-grade system with a React/Next.js frontend, DynamoDB storage, campaign journal, Stripe payments, and enhanced AI features.

## Code Location

- All Phase 2 code lives in: `WinFlip Phase 2 - MVP/`
- Frontend: `WinFlip Phase 2 - MVP/frontend/` (Next.js)
- Backend: `WinFlip Phase 2 - MVP/backend/` (Lambda functions + CDK)
- Phase 1 reference code: `Project Icarus Phase 1 - POC/Icarus/`
- Spec files: `.kiro/specs/winflip-campaign-journal-mvp/`

## Package Management

- Always use `uv` as the Python package installer. Never use `pip`.
- Use `uv add` to add dependencies, `uv sync` to install, `uv run` to execute.
- Each Lambda function has its own `pyproject.toml` for isolated dependencies.
- Frontend uses `npm` for Node.js dependencies.

## Tech Stack

### Frontend
- React / Next.js (App Router)
- TypeScript
- AWS Amplify for hosting and CI/CD
- Amplify Auth library for Cognito integration
- TanStack Query (React Query) for server state
- React Context + useReducer for global state (auth, settings)

### Backend
- Python 3.13 for all Lambda functions
- Pydantic for data models and validation
- boto3 for AWS service interactions
- Amazon Bedrock (Claude Sonnet) for all AI operations
- ULID for sortable unique IDs

### Infrastructure
- AWS CDK (TypeScript) for infrastructure-as-code
- DynamoDB for structured data (questionnaires, journal entries, insights, actions, chat sessions, subscriptions)
- S3 for unstructured data (election data, prompts, generated insight documents)
- API Gateway (REST) for all API endpoints
- Amazon Cognito for authentication
- CloudWatch for logging and metrics
- Stripe (sandbox mode) for payments

## Key Conventions

### Python / Lambda
- Use Pydantic models for all request/response schemas in `backend/lambdas/shared/models.py`
- Use the shared `logger.py` module for structured JSON logging to CloudWatch
- Use the shared `dynamo.py` module for DynamoDB operations
- Use the shared `bedrock.py` module for Bedrock client calls
- All Lambda handlers follow the pattern: parse request → validate → process → return response
- Error responses use a consistent `error_response(status_code, message)` helper
- Environment variables for all configurable values (bucket names, model IDs, table names)
- Temperature 0.3 for chatbot/insights, 0.1 for entry tagger (deterministic classification)

### TypeScript / Frontend
- Use Next.js App Router (`src/app/` directory structure)
- Components in `src/components/` organized by feature (journal/, chatbot/, insights/, etc.)
- API client in `src/lib/api.ts` as a fetch wrapper
- Type definitions in `src/lib/types.ts`
- Custom hooks in `src/hooks/` (useAuth, useJournal, useChatbot)
- WinFlip branding tokens in `src/styles/globals.css`

### CDK / Infrastructure
- Main stack: `WinFlipStack` in `backend/infra/lib/winflip-stack.ts`
- Modular constructs: `api-construct.ts`, `auth-construct.ts`, `storage-construct.ts`, `lambda-construct.ts`
- Environment-aware deployment via CDK context (`--context env=dev` or `--context env=prod`)
- S3 bucket naming: `winflip-{purpose}-{accountId}`
- DynamoDB table naming: `winflip-{entity}` (e.g., `winflip-journal-entries`)
- Lambda naming: `winflip-{function}-lambda`

## DynamoDB Patterns

- Partition key is always `candidateId` (Cognito user sub)
- Sort key is entity-specific (entryId, insightId, actionId, sessionId)
- Use ULID for sort keys (time-sortable)
- GSIs for tag-based and date-range queries on journal entries
- All timestamps in ISO 8601 format

## AI / Bedrock Patterns

- Prompt templates stored in S3 `winflip-prompts-{accountId}` bucket
- Prompts use `{placeholder}` syntax for variable substitution
- Token limit handling: estimate tokens first, chunk if needed, summarize partial results
- Entry tagger returns structured JSON with `location_tag`, `topic_tag`, `event_type_tag`
- Archetype tone levels: Cautious, Calm, Balanced, Engaged, Fired-up
- Default tone: Balanced

## API Design

- All endpoints require Cognito JWT auth except `/stripe/webhook`
- Stripe webhook validates `Stripe-Signature` header
- REST conventions: GET for reads, POST for creates/actions, PUT for updates
- All responses include `Content-Type: application/json` and `Access-Control-Allow-Origin` headers
- Error responses: `{ "error": "message" }` with appropriate HTTP status codes

## Testing

- Unit tests in `backend/tests/unit/`
- Property-based tests in `backend/tests/property/` (serialization round-trips, data model invariants)
- Frontend tests alongside components
- Use `uv run pytest` for backend tests
- Use `npm run test -- --run` for frontend tests (single execution, not watch mode)

## Migration Notes

- Phase 1 `PCMChatbot` class is refactored into `backend/lambdas/insights/pcm_chatbot.py`
- Phase 1 `trigger_chatbot.py` and `check_LLM_response_lambda.py` are removed (replaced by synchronous pattern)
- Phase 1 S3 bucket names (`icarus-*`) become `winflip-*`
- Phase 1 CDK stack `IcarusDannerInfraStack` becomes modular `WinFlipStack`
- Election data and prompt templates are preserved and copied to new bucket names

## Out of Scope

- Comet integration (explicitly removed)
- Multi-state data beyond Virginia
- Full production HA/DR
- Native mobile apps (iOS/Android)
- Real-time social media monitoring
- Full NGP VAN / L2 integration
- Complex subscription management (cancel/downgrade) — MVP is free→paid upgrade only
