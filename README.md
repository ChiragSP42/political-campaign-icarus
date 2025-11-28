# 🎯 Project Icarus
## A Political Campaign Assistant Chatbot

> *Democratizing campaign intelligence. Because every candidate deserves a war room.*

---

## 🌟 What is Project Icarus?

Political campaigns have always been won by those with the deepest pockets and the largest teams. Well-funded operations employ entire squads of data analysts, strategists, and researchers working around the clock to synthesize electoral trends, identify voter patterns, and craft winning strategies.

**Project Icarus changes that.**

This platform collapses an entire analytics department into a conversational AI assistant—your personal strategic advisor, available 24/7. Whether you're a grassroots candidate with a shoestring budget or a seasoned political operator looking to sharpen your edge, Icarus gives you institutional-grade insights and real-time strategic guidance that rivals those managed by campaigns with multimillion-dollar war chests.

---

## 🏗️ The Dual-Engine Architecture

Project Icarus operates on two interconnected systems, each designed to solve distinct strategic challenges:

### 🔍 Engine 1: The Insights Powerhouse

Your campaign doesn't operate in a vacuum. Every voter, every precinct, and every election tells a story encoded in data. The **Insights Engine** decodes that story by conducting forensic-level analysis of five years of historical election data, aggregated down to individual precinct granularity.

Here's what makes it revolutionary:

- **Hyper-localized intelligence**: Understand voting patterns street-by-street, precinct-by-precinct. No generic national trends—just *your* neighborhood, *your* voters.
- **Personality-calibrated strategy**: Your initial questionnaire isn't just flavor. It feeds directly into an AI system that personalizes every recommendation to *your* political positioning and candidate profile.
- **Five comprehensive analytical lenses**:
  - 📍 **Precinct-Level Forensics**: Drill into individual precincts and identify geographic strongholds and weakness zones
  - 👥 **Demographic Intelligence**: Map constituencies to voting behaviors, turnout patterns, and issue alignment
  - 🗳️ **Turnout Scenarios**: Model how changes in voter participation reshape the electoral landscape
  - ⚔️ **Competitive Positioning**: See how you stack up against opponents across every measurable dimension
  - 🎯 **Winning Tactics**: Data-driven strategic recommendations forged from real electoral patterns

### 💬 Engine 2: Your AI Strategic Partner

Insights reports are powerful, but static. Enter the **Interactive Chatbot**—a conversational layer that transforms analysis into strategy.

This isn't a basic Q&A bot. It maintains full context awareness of your generated insights, has been trained on comprehensive election law and regulatory frameworks, and understands the nuances of your specific electoral landscape. Ask follow-up questions that drill deeper into specific metrics. Explore hypothetical scenarios ("What if turnout increases 15%?"). Get guidance on electoral compliance. The chatbot bridges the chasm between high-level strategy and operational execution.

---

## 🚀 Installation & Local Development

Ready to set up your personal campaign command center? Let's walk through this carefully.

### 📋 Prerequisites

This project assumes you're comfortable with Python, AWS infrastructure, and modern development workflows. You'll need basic familiarity with the command line and package management.

### 🔧 Phase 1: Python Environment Isolation

Clean dependency management is *critical* for avoiding system Python conflicts that can derail development. We use **uv**, a blazingly fast Python package manager that handles both dependency resolution and virtual environment orchestration with elegance.

#### Step 1️⃣ : Install uv

Head over to the [official uv documentation](https://docs.astral.sh/uv/getting-started/installation/) and follow the installation instructions for your operating system. This typically takes just a few minutes and provides substantial performance improvements over traditional pip.

```bash
# Verify installation
uv --version
```

#### Step 2️⃣ : Create Your Isolated Virtual Environment

From the project root directory, spin up a fresh Python 3.13 environment:

```bash
uv venv --python 3.13
```

This command creates an isolated Python environment in `.venv/`, keeping your project dependencies completely sandboxed from your system Python installation. This is *essential* for reproducible deployments.

#### Step 3️⃣ : Activate the Virtual Environment

Your shell matters here. Choose the right command for your system:

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
.venv\Scripts\activate
```

After activation, your shell prompt should display `(.venv)` as a prefix. This confirms the environment is active and all Python commands will use your isolated environment.

#### Step 4️⃣ : Install Project Dependencies

```bash
uv add -r requirements.txt
```

This single command reads `requirements.txt` and installs all dependencies with locked versions, ensuring reproducible builds across machines and environments. No more "works on my machine" surprises.

---

### 🎨 Phase 2: Launching the Streamlit Frontend

The user-facing interface is built with **Streamlit**, a Python framework for building data applications with minimal boilerplate. The frontend provides an intuitive, interactive interface for uploading your personality profile, exploring insights, and engaging with the chatbot in real time.

#### Launch the Application

Navigate to the `icarus-streamlit` directory:

```bash
cd icarus-streamlit
python3 -m streamlit run streamlit_app.py
```

Streamlit will start a development server (typically at `http://localhost:8501`) and automatically open your default browser. The hot-reload functionality means code changes are instantly reflected—perfect for rapid iteration and debugging.

**Pro tip**: Streamlit caches computations automatically, so subsequent runs are blazing fast.

---

## ☁️ AWS Infrastructure Deployment

Project Icarus leverages AWS cloud infrastructure for enterprise-grade scalability, reliability, and security. The infrastructure-as-code approach using **AWS CDK** with TypeScript ensures your deployment is reproducible, version-controlled, auditable, and maintainable.

### 🔐 Phase 1: Dependency Installation

Before running any CDK commands, you need to install three critical tools: AWS CLI (authentication gateway), npm (package manager), and AWS CDK CLI (infrastructure orchestrator).

#### Step 1️⃣ : Install AWS CLI

Follow the [official AWS CLI installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html#getting-started-install-instructions). The AWS CLI provides the `aws` command for programmatic AWS interactions and authentication.

```bash
# Verify installation
aws --version
```

#### Step 2️⃣ : Install Node Package Manager (npm)

npm comes bundled with **Node.js**. Download and install from the [official Node.js website](https://nodejs.org/). This gives you `npm` for managing JavaScript and TypeScript dependencies.

```bash
# Verify installation
npm --version
node --version
```

#### Step 3️⃣ : Install AWS CDK CLI

Once npm is installed, install the AWS CDK command-line tool globally:

```bash
npm install -g aws-cdk
```

Verify the installation:

```bash
cdk --version
```

#### Step 4️⃣ : Configure AWS Credentials

The AWS CLI needs credentials to authenticate with your AWS account. You have two approaches:

**Option A: Default Profile** (simpler, ideal for single-account workflows)
```bash
aws configure
```

**Option B: Named Profile** (recommended for multi-account setups—this is what we prefer)
```bash
aws configure --profile <your-profile-name>
```

You'll be prompted for:
- **AWS Access Key ID**: Your IAM user's access key
- **AWS Secret Access Key**: Your IAM user's secret key (store this securely!)
- **Default region**: Region where resources will be deployed (e.g., `us-east-1`, `us-west-2`)
- **Default output format**: `json` is recommended for machine parsing

These credentials are encrypted and stored in `~/.aws/credentials` and `~/.aws/config` for future use.

---

### 🏗️ Phase 2: AWS Infrastructure Provisioning

This is where the magic happens. We're about to create your cloud-based campaign command center.

#### Step 1️⃣ : Navigate to the CDK Directory

```bash
cd icarus-cdk/infra
```

This directory contains your TypeScript infrastructure-as-code that defines all AWS resources.

#### Step 2️⃣ : Bootstrap Your AWS Account (One-Time Setup)

Before deploying CDK stacks to a new AWS account or region combination, CDK requires a one-time bootstrap operation. This creates foundational CloudFormation resources and S3 buckets that CDK uses for artifact staging:

```bash
cdk bootstrap --profile <your-profile-name>
```

This command is **idempotent** (safe to run multiple times). It creates:
- A CloudFormation stack named `CDKToolkit`
- An S3 bucket for storing CDK artifacts during deployment
- IAM roles and permissions that allow CDK to manage resources
- Other necessary infrastructure components

**Note**: This is a one-time operation per AWS account per region. If you deploy to multiple regions, you'll need to bootstrap each region separately.

#### Step 3️⃣ : Deploy Your Infrastructure

```bash
cdk deploy --profile <your-profile-name>
```

This command synthesizes your TypeScript CDK code into a CloudFormation template and deploys it to AWS. You'll see:

1. A detailed diff showing exactly what resources will be created
2. A prompt asking for confirmation before proceeding (type `y` to confirm)
3. Real-time deployment progress as CloudFormation creates resources

**Critical Concept**: `cdk deploy` creates only the **infrastructure skeleton**. Think of it as constructing an empty building with all necessary walls, plumbing, electrical wiring, and HVAC systems. It does *not* populate content within that infrastructure. You'll subsequently need to:
- Populate S3 buckets with election data
- Configure authentication credentials and email allowlists
- Set up knowledge bases and vector stores
- Initialize databases with seed data

These content provisioning steps happen through separate processes (like the web scraper and data pipeline scripts).

#### Step 4️⃣ : Destroying Infrastructure (Cleanup)

When you're done or want to start fresh, tear down all AWS resources to avoid unnecessary charges:

```bash
cdk destroy --profile <your-profile-name>
```

CloudFormation will ask for confirmation before deleting resources. This operation removes all stacks created by your CDK deployment.

⚠️ **Use with caution**: This operation is not easily reversible, and data deletion may be permanent depending on your resource configurations.

---

## 📊 Data Ingestion: Election Data Scraping

Project Icarus derives its analytical power from comprehensive, granular electoral data. The system includes an automated web scraper that ingests five years of election data from the Virginia Department of Elections, covering all office types, precincts, and electoral contests.

### Running the Web Scraper

```bash
uv run web_scraper.py
```

This command orchestrates a sophisticated data pipeline that:

1. **Connects to official sources**: Establishes secure connections to Virginia Department of Elections public data endpoints
2. **Intelligently scrapes data**: Retrieves election results for all available offices, precincts, and candidates across the past five years
3. **Normalizes and validates**: Transforms raw data into consistent formats with validation to catch anomalies
4. **Ingests into knowledge base**: Stages data for integration into your Bedrock knowledge base and vector store
5. **Handles edge cases gracefully**: Manages pagination, implements rate limiting, and recovers from transient failures

The scraper runs with built-in progress indicators and comprehensive error logging. Depending on data volume and network conditions, this process may take several minutes. Monitor console output for progress updates and any data validation warnings.

---

## 🎯 Next Steps

Your Icarus instance is now operational. Here's what's next:

- **Load your personality profile** into the Streamlit interface to calibrate strategic recommendations
- **Run the web scraper** to populate your election data knowledge base
- **Start asking questions** through the chatbot interface
- **Monitor campaign performance** through real-time analytics dashboards
- **Iterate and refine** your strategy based on insights and feedback

Welcome to the future of campaign intelligence. 🚀

---

## 📚 Additional Resources

For deeper dives into specific components:

- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [AWS CLI Configuration Guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
- [Python uv Documentation](https://docs.astral.sh/uv/)

---

**Built with precision. Designed for victory.** ⚡