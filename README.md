# 🗳️ WinFlip
## AI-Powered Political Campaign Strategy Platform

> *Democratizing campaign intelligence. Because every candidate deserves a war room.*

---

## What is WinFlip?

WinFlip collapses an entire campaign analytics department into a conversational AI assistant — a personal strategic advisor, available 24/7. Whether you're a grassroots candidate with a shoestring budget or a seasoned political operator, WinFlip delivers institutional-grade insights and real-time strategic guidance.

Built for Virginia elections, the platform ingests 5+ years of precinct-level election data, election law knowledge bases, and candidate-specific context to generate comprehensive campaign strategies and answer strategic questions in real time.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          AWS Cloud (us-east-1)                          │
│                                                                         │
│  ┌───────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Cognito     │    │ API Gateway  │    │     S3 Buckets           │  │
│  │  User Pool    │    │  (REST API)  │    │  - election-data         │  │
│  │  (email auth) │    │  /chat       │    │  - prompt-bucket         │  │
│  └───────────────┘    │  /sessions   │    │  - election-laws         │  │
│                       │  /save-quest │    │  - chatbot-responses     │  │
│                       │  /check-*    │    └──────────────────────────┘  │
│                       └──────┬───────┘                                  │
│                              │                                          │
│  ┌───────────────────────────┼───────────────────────────────────────┐  │
│  │                    Lambda Functions                               │  │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐   │  │
│  │  │ trigger_chatbot  │  │ chatbot_lambda   │  │ generate_      │   │  │
│  │  │ (async invoke)   │→ │ (Docker, 2GB)    │  │ insights_v2    │   │  │
│  │  └──────────────────┘  │ Bedrock Converse │  │ (Docker, 2GB)  │   │  │
│  │                        │ + KB retrieval   │  │ N+1 strategy   │   │  │
│  │  ┌─────────────────┐   └──────────────────┘  └───────┬────────┘   │  │
│  │  │ check_LLM_resp  │                                 │            │  │
│  │  │ (poll for reply)│                          DynamoDB Streams    │  │
│  │  └─────────────────┘                                 │            │  │
│  │  ┌──────────────────┐  ┌───────────────────┐  ┌──────┴─────────┐  │  │
│  │  │ session_manager  │  │ save_questionnaire│  │ Questionnaire  │  │  │
│  │  │ (CRUD sessions)  │  │ (→ DynamoDB)      │  │ Table (DDB)    │  │  │
│  │  └──────────────────┘  └───────────────────┘  └────────────────┘  │  │
│  │  ┌───────────────────┐                                            │  │
│  │  │check_questionnaire│                                            │  │
│  │  └───────────────────┘                                            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  DynamoDB Tables  │  │  RDS PostgreSQL  │  │  Bedrock             │  │
│  │  - Main           │  │  virginia_       │  │  - Claude Sonnet     │  │
│  │  - Chat History   │  │    elections     │  │  - Knowledge Base    │  │
│  │  - Questionnaire  │  │  (precinct-level │  │    (election laws)   │  │
│  │                   │  │   election data) │  │                      │  │
│  └───────────────────┘  └──────────────────┘  └──────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Frontend: Next.js on AWS Amplify                                │   │
│  │  - Auth (Cognito)  - Questionnaire  - Dashboard (chat+insights)  │   │
│  │  - Journal pages   - Markdown rendering  - Draggable split-panel │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── icarus-ui/                    # Next.js frontend (deployed on Amplify)
├── icarus-cdk/
│   ├── infra/                    # CDK stack (TypeScript) — all AWS resources
│   ├── services/lambdas/         # Lambda function source code
│   └── local-files/              # Prompt templates & election cycle config
├── election-data/                # 43 Virginia election JSON files (local)
├── election-data-migration/      # ETL pipeline: JSON → PostgreSQL on RDS
├── prototype_chatbot_backend/    # Early CLI-based chatbot prototype
├── aws_helpers/                  # Shared Python utilities (S3, logging, etc.)
├── web_scraper.py                # Tavily-powered election data scraper
├── MIT_scraper.py                # Harvard Dataverse precinct data puller
├── strands_scratch.py            # Strands Agent experiment (SQL tool-use)
├── election_cycles.json          # Virginia election cycle definitions (20 offices)
├── amplify.yml                   # Amplify build config for Next.js
├── .github/workflows/deploy.yml  # GitHub Actions CI/CD for CDK
└── env-example.txt               # All environment variables documented
```

---

## How It Works

1. **Sign up & verify** — User creates an account via Cognito and confirms their email.
2. **Questionnaire** — Candidate fills out an intake form (office, district, background, communication style). Answers are saved to the Questionnaire DynamoDB table.
3. **Insight generation** — A DynamoDB Streams trigger fires the `generate-insights-lambda`, which pulls 5 years of precinct-level election data from S3, retrieves election laws from a Bedrock Knowledge Base, and generates a comprehensive strategic analysis via Claude. If the data exceeds the 190K input token limit, an N+1 chunking strategy splits the data across concurrent Bedrock calls and stitches the results.
4. **Dashboard** — Split-panel view with chat on the left and rendered insights on the right (resizable draggable divider). If insights aren't ready yet, the UI polls automatically.
5. **Chat** — Conversational AI that has full context of the candidate's insights, questionnaire answers, election data, and regulatory framework. Conversation history is persisted in DynamoDB and loaded on each turn.

---

## What Was Built & How (XYZ Format)

### Data Collection & Ingestion

- **Built a Tavily-powered web scraper** (`web_scraper.py`) by crawling Virginia's historical elections portal, extracting precinct-level CSV data, computing win/flip numbers per precinct, and storing structured JSON files in S3 — so that the platform has 5+ years of granular election data across 20 office types.

- **Built a Harvard Dataverse scraper** (`MIT_scraper.py`) by querying the MIT Election Data Science Lab's API (DOI: `10.7910/DVN/NT66Z3`), downloading Virginia precinct returns, and parsing TSV/CSV formats — so that we have an alternative, academically sourced dataset for validation and enrichment.

- **Defined a Virginia election cycle configuration** (`election_cycles.json`) by mapping 20 office types (President through Town Council) with their cycle lengths, election patterns (even/odd/biennial/periodic), and statewide flags — so that the scraper and insight generator know exactly which years to look back for each office.

### Election Data Migration (JSON → PostgreSQL)

- **Built a one-time ETL pipeline** (`election-data-migration/`) by writing a Python script (`etl.py`) that discovers 43 JSON files from local disk or S3, validates their structure, and batch-inserts elections → districts → precincts → results into a normalized PostgreSQL schema on RDS — so that election data is queryable via SQL instead of scanning JSON files.

- **Designed a normalized relational schema** (`schema.sql`) with four tables (`elections`, `districts`, `precincts`, `results`), foreign key cascades, performance indexes on all lookup columns, and a `pg_trgm` trigram index for fuzzy precinct name search — so that queries against 43 elections worth of precinct data are fast and flexible.

- **Made the ETL idempotent** by checking each file's `record_id` against the `elections` table before inserting and wrapping each file in its own transaction — so that the script is safe to re-run without duplicating data or losing progress on partial failures.

- **Automated the ETL on EC2** by adding a CDK-managed EC2 instance (gated behind `DEPLOY_ETL_EC2=true` env var) that boots, installs dependencies, fetches RDS credentials from Secrets Manager, downloads `etl.py` and `schema.sql` from S3, runs the schema then the ETL in S3 mode, uploads logs, and shuts itself down — so that the migration runs hands-free in the cloud without needing a local machine with DB access.

- **Known issues**: Two files failed migration due to `float NaN` values — `President/2024/General_Election` and `Town_Council/2021/Special_General_Election`.

### CDK Infrastructure (`icarus-cdk/infra/`)

- **Provisioned the full backend via a single CDK stack** (`IcarusDannerInfraStack`) by defining Cognito, API Gateway, 7 Lambda functions, 3 DynamoDB tables, 3 S3 buckets, a VPC, an RDS PostgreSQL instance, and an optional ETL EC2 instance — all in one TypeScript file — so that the entire backend can be deployed or torn down with a single `cdk deploy` / `cdk destroy`.

- **Set up Cognito authentication** with email-based sign-in, auto-verification, password policy (8+ chars, upper/lower/digits), and a web client with SRP auth flows — so that users can sign up, verify, and log in securely.

- **Created three DynamoDB tables**: Main (user profiles + insights, PK: `userId`, SK: `SK`), Questionnaire (PK: `userId`, with DynamoDB Streams enabled), and Chat History (PK: `chatId`, SK: `timestamp`, with a `userId-index` GSI) — so that user data, questionnaire answers, and conversation history are stored with pay-per-request billing.

- **Deployed the chatbot and insights lambdas as Docker images** (2GB memory, 15-min timeout, ECR-backed) — so that large dependencies (tiktoken, psycopg2, etc.) fit within Lambda's deployment limits.

- **Wired DynamoDB Streams** from the Questionnaire table to the `generate-insights-lambda` — so that insight generation triggers automatically the moment a candidate submits their questionnaire, with zero polling.

- **Provisioned RDS PostgreSQL 16** (t3.micro, publicly accessible, 20–50GB auto-scaling storage) in a dedicated VPC with public subnets and a security group allowing port 5432 — so that the election data is stored in a relational database accessible for both the ETL and future Strands agent queries.

- **Set up API Gateway** with CORS, throttling (1000 req/s, 2000 burst), and six routes (`/chat`, `/check-response`, `/save-questionnaire`, `/check-questionnaire`, `/sessions`, `/sessions/messages`) — so that the frontend can communicate with all backend services through a single REST endpoint.

### Lambda Services

- **Built `save_questionnaire_lambda`** to receive questionnaire answers from the frontend and persist them in the Questionnaire DynamoDB table — so that candidate data is durably stored and triggers downstream insight generation via Streams.

- **Built `generate_insights_lambda_v2`** (Docker) to consume DynamoDB Streams events, load the candidate's questionnaire, pull 5 years of precinct-level election data from S3, retrieve election laws from a Bedrock Knowledge Base, format everything into a prompt, and call Claude via the Converse API — so that comprehensive campaign insights are generated automatically. Implemented an **N+1 chunking strategy** with concurrent `ThreadPoolExecutor` calls to handle cases where election data exceeds the 190K input token limit, followed by a final "stitching" LLM call to merge chunk outputs into a cohesive document.

- **Built `chatbot_lambda`** (Docker) to load conversation history from DynamoDB, inject the candidate's insights and questionnaire into a system prompt, and call Claude via the Converse API — so that the chat is context-aware across sessions and grounded in the candidate's actual strategic analysis.

- **Built `trigger_chatbot`** as an async invocation wrapper that writes the user message to DynamoDB and invokes the chatbot lambda asynchronously — so that the frontend gets an immediate response and can poll for the result.

- **Built `check_LLM_response_lambda`** to poll DynamoDB for the assistant's response by `chatId` — so that the frontend can check when the chatbot has finished generating its reply.

- **Built `session_manager_lambda`** to handle CRUD operations for chat sessions (list, get messages, delete) — so that users can manage multiple conversation threads.

### Frontend (`icarus-ui/`)

- **Built a Next.js 15 frontend** with server-side rendering, Tailwind CSS, and TypeScript, deployed on AWS Amplify — so that the platform has a fast, modern web interface with automatic CI/CD on push to `main`.

- **Implemented Cognito authentication** with cookie-based sessions and token validation (migrated from localStorage) — so that auth state persists across tabs and is more secure.

- **Built a candidate questionnaire page** that collects office, district, background, and communication style — so that the platform has the context needed to generate personalized insights.

- **Built a dashboard with a draggable split-panel layout** — chat on the left, rendered insights (Markdown) on the right, with a resizable divider — so that candidates can reference their strategic analysis while chatting with the AI.

- **Added automatic insight polling** — after questionnaire submission, the UI waits ~5 minutes and then checks if insights are ready; on the dashboard, the insights panel auto-refreshes — so that candidates don't have to manually reload.

- **Built a chat interface** with session management (create, switch, delete sessions), collapsible session sidebar, and Markdown rendering of LLM responses — so that conversations are organized and readable.

- **Built Journal pages** (Overview, Insights, Actions, Entries) with sub-navigation — so that there's a structured space for campaign tracking beyond the chat interface.

- **Renamed the project from "Icarus" to "WinFlip"** across the UI — so that the branding reflects the product's identity.

### CI/CD & Deployment

- **Set up GitHub Actions CI/CD** (`.github/workflows/deploy.yml`) with OIDC-based AWS credential exchange — so that pushing to `winflip` or `winflip-dev` branches automatically deploys the CDK stack without storing long-lived AWS keys.

- **Configured Amplify build** (`amplify.yml`) to inject environment variables into `.env.production` at build time and run from the `icarus-ui/` app root — so that the frontend deploys correctly with all required config.

### DynamoDB Migration (from S3)

- **Migrated the data layer from S3 to DynamoDB** by shifting questionnaire storage, insight storage, and chat history from S3 JSON files to three DynamoDB tables — so that reads/writes are faster, atomic, and don't require S3 list/get round-trips. This was done across the `feature/ddb-migration` branch and merged into `winflip-dev`.

### Token Limit Handling

- **Solved the Bedrock input token ceiling** by implementing an N+1 strategy in the insights lambda: count tokens with tiktoken, split election data into N chunks if over 190K tokens, run concurrent Bedrock calls (up to 3 workers), and stitch results with a final polishing LLM call — so that candidates running for offices with large datasets (e.g., House of Delegates District 1) still get complete insights.

### Strands Agent Experiment (In Progress)

- **Started building a Strands-based SQL agent** (`strands_scratch.py`) by defining `@tool`-decorated functions (`election_record`, `district_record`) that query the PostgreSQL RDS database and wiring them into a `strands.Agent` — so that the chatbot can eventually answer questions by querying structured election data directly from the database instead of relying solely on S3 JSON files.

---

## Local Development

### Prerequisites
- Node.js 22+ and npm
- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- AWS CLI configured with credentials
- AWS CDK CLI (`npm install -g aws-cdk`)

### Frontend (Next.js)

```bash
cd icarus-ui
cp .env.local.example .env.local   # Fill in your values
npm install
npm run dev
```

### CDK Infrastructure

```bash
cd icarus-cdk/infra
npm install

# First time only — bootstrap CDK in your AWS account
cdk bootstrap --profile <your-profile>

# Deploy
cdk deploy --profile <your-profile>
```

### Python Dependencies (for Lambda development)

```bash
uv venv --python 3.13
source .venv/bin/activate
uv add -r aws_helpers/requirements.txt
```

---

## Environment Variables

Copy `env-example.txt` to `.env` in the project root and fill in values. This file is loaded by the CDK stack via dotenv.

For the frontend, copy `icarus-ui/.env.local.example` to `icarus-ui/.env.local`.

Key variables:

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY` / `AWS_SECRET_KEY` | AWS credentials for scrapers and local scripts |
| `CUSTOM_ACCESS_KEY_ID` / `CUSTOM_SECRET_ACCESS_KEY` | AWS credentials for Next.js server-side API routes |
| `COGNITO_USER_POOL_ID` / `COGNITO_CLIENT_ID` | From CDK outputs |
| `API_ENDPOINT` | API Gateway URL from CDK outputs |
| `MAIN_TABLE_NAME` | DynamoDB Main table name (`main-<ACCOUNT_ID>`) |
| `MODEL_ID` | Bedrock model ID (default: Claude Sonnet) |
| `KB_ID` | Bedrock Knowledge Base ID (election laws) |
| `TAVILY_API_KEY` | Tavily API key for web scraping |
| `DEPLOY_ETL_EC2` | Set to `true` to include the one-time ETL EC2 instance in CDK deploy |
| `DB_HOST` / `DB_NAME` / `DB_USERNAME` / `DB_PASSWORD` / `DB_PORT` | PostgreSQL RDS connection details |

See `env-example.txt` for the full list with documentation.

---

## Deployment

### Frontend (Amplify)
Push to `main` branch. Amplify auto-builds and deploys. Environment variables must be configured in the Amplify console and included in the `amplify.yml` grep pattern.

### Infrastructure (GitHub Actions)
The `.github/workflows/deploy.yml` workflow runs `cdk deploy` on push to `winflip` or `winflip-dev` branches using OIDC-based AWS credentials.

---

## Election Data Migration

See [`election-data-migration/README.md`](election-data-migration/README.md) for the full step-by-step guide to deploying the RDS instance, running the schema, and executing the ETL.

---

## Cleanup

```bash
cd icarus-cdk/infra
cdk destroy --profile <your-profile>
```

⚠️ This deletes all AWS resources including DynamoDB tables and the RDS instance. Data deletion is permanent.
