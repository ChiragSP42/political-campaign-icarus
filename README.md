# 🗳️ WinFlip
## AI-Powered Political Campaign Strategy Platform

> *Democratizing campaign intelligence. Because every candidate deserves a war room!*

---

## What is WinFlip?

WinFlip collapses an entire campaign analytics department into a conversational AI assistant — your personal strategic advisor, available 24/7. Whether you're a grassroots candidate with a shoestring budget or a seasoned political operator, WinFlip delivers institutional-grade insights and real-time strategic guidance.

---

## Architecture Overview

The platform consists of three main components:

### Next.js Frontend (`icarus-ui/`)
A server-rendered React application deployed on AWS Amplify. Handles authentication (Cognito), the candidate questionnaire, an insights dashboard, and an interactive chat interface with a draggable split-panel layout.

### AWS CDK Infrastructure (`icarus-cdk/infra/`)
TypeScript CDK stack that provisions all backend resources:
- **Cognito** — User authentication (email/password with verification)
- **API Gateway** — REST API for chat, questionnaire, and session management
- **Lambda Functions** (Python 3.13) — Seven lambdas handling chatbot logic, questionnaire storage, insight generation, session management, and response checking
- **DynamoDB** — Three tables: Main (user profiles + insights), Questionnaire, and Chat History
- **S3** — Election data, prompt templates, and chatbot responses
- **Bedrock** — Claude model for AI-powered insights and chat

### Lambda Services (`icarus-cdk/services/lambdas/`)
- `save_questionnaire_lambda` — Stores candidate questionnaire answers in DynamoDB
- `generate_insights_lambda` — Triggered by DynamoDB Streams when a questionnaire is saved; generates comprehensive campaign insights using Bedrock and stores them in the Main table
- `chatbot_lambda` — Main conversational AI with full context awareness of generated insights and election law knowledge base
- `trigger_chatbot` — Async invocation wrapper for the chatbot
- `check_LLM_response_lambda` — Polls for completed chatbot responses
- `session_manager_lambda` — CRUD operations for chat sessions
- `check_questionnaire_lambda` — Checks if a user has completed the questionnaire

---

## How It Works

1. **Sign up & verify** — User creates an account via Cognito, confirms email
2. **Questionnaire** — Candidate fills out an intake form (office, district, background, communication style)
3. **Insight generation** — DynamoDB Streams triggers the insights lambda, which pulls 5 years of precinct-level election data from S3, retrieves election laws from a Bedrock Knowledge Base, and generates a comprehensive strategic analysis via Claude
4. **Dashboard** — Split-panel view with chat on the left and rendered insights on the right (resizable divider)
5. **Chat** — Conversational AI that has full context of the candidate's insights, election data, and regulatory framework

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

For the frontend, copy `icarus-ui/.env.local.example` to `icarus-ui/.env.local`. See `env-example.txt` for documentation on all required variables.

Key variables:
- `CUSTOM_ACCESS_KEY_ID` / `CUSTOM_SECRET_ACCESS_KEY` — AWS credentials for server-side API routes
- `COGNITO_USER_POOL_ID` / `COGNITO_CLIENT_ID` — From CDK outputs
- `API_ENDPOINT` — API Gateway URL from CDK outputs
- `MAIN_TABLE_NAME` — DynamoDB Main table name (`main-<ACCOUNT_ID>`)
- `MODEL_ID` — Bedrock model ID
- `KB_ID` — Bedrock Knowledge Base ID

---

## Deployment

### Frontend (Amplify)
Push to `main` branch. Amplify auto-builds and deploys. Environment variables must be configured in the Amplify console and included in the `amplify.yml` grep pattern.

### Infrastructure (GitHub Actions)
The `.github/workflows/deploy.yml` workflow runs `cdk deploy` on push to `main` using OIDC-based AWS credentials.

---

## Cleanup

```bash
cd icarus-cdk/infra
cdk destroy --profile <your-profile>
```

⚠️ This deletes all AWS resources including DynamoDB tables. Data deletion is permanent.
