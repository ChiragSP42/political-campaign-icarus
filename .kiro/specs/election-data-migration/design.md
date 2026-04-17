# Design Document: Election Data Migration

## Overview

This design covers the one-time migration of 43 Virginia election JSON files (2020–2025) into a PostgreSQL database on AWS RDS. The JSON files follow a nested hierarchy — Election → Districts → Precincts → Results — that must be flattened into a normalized relational schema.

The system consists of three parts:
1. A CDK construct (TypeScript) added to the existing `icarus-cdk/infra/lib/infra-stack.ts` that provisions a VPC, security group, and RDS PostgreSQL instance with credentials in Secrets Manager.
2. A PostgreSQL schema with 4 tables (`elections`, `districts`, `precincts`, `results`) plus indexes and the `pg_trgm` extension for fuzzy precinct name search.
3. A Python ETL script that reads JSON files (local or S3), parses and normalizes the data, and loads it into the database with per-file transaction safety and idempotency via `record_id` uniqueness.

### Key Design Decision: District-Level vs Precinct-Level Results

The JSON contains both `district_results[]` (aggregate candidate totals per district) and `precinct.results[]` (candidate totals per precinct). The district-level results are mathematically derivable from precinct sums, but we **store both** for these reasons:

- **Query performance**: Avoids expensive `SUM() ... GROUP BY` aggregations across thousands of precinct rows for common queries like "who won District_100?"
- **Data fidelity**: The source data provides both levels explicitly — storing both preserves the original data exactly as published, which matters if rounding or special precincts cause the sum to not match the official district total.
- **Schema simplicity**: The `results` table uses a polymorphic foreign key pattern (`district_id` OR `precinct_id`, one NULL) to associate results with their parent level. This is simpler than maintaining a separate materialized view.

## Architecture

```mermaid
flowchart TD
    A[JSON Files<br/>local or S3] --> B[ETL Script<br/>Python + psycopg2]
    B --> C{Source Mode}
    C -->|--source local| D[Recursive file discovery<br/>election-data/]
    C -->|--source s3| E[S3 list + download<br/>predictif-election-data bucket]
    D --> F[Parse & Validate JSON]
    E --> F
    F --> G[Transform & Normalize]
    G --> H[Load into PostgreSQL<br/>per-file transactions]
    H --> I[(AWS RDS PostgreSQL<br/>virginia_elections)]
    
    subgraph "Per-File Transaction"
        G1[Insert election] --> G2[Insert districts + district results]
        G2 --> G3[Insert precincts + precinct results]
    end
```

### ETL Flow Per File

1. **Read**: Load JSON from disk or S3
2. **Validate**: Check required top-level fields (`record_id`, `year`, `office`, `stage`, `total_votes`, `districts`)
3. **Idempotency check**: Query `elections` table for existing `record_id` — skip if found
4. **Transform**: Parse precinct names, normalize optional fields, convert float votes to integers
5. **Load**: Insert `election` → `districts` + `district_results` → `precincts` + `precinct_results` inside a single transaction
6. **Commit or rollback**: Commit on success, rollback on any error for that file

## Components and Interfaces

### 0. CDK Infrastructure (TypeScript)

The RDS instance is provisioned via CDK within the existing `icarus-cdk/infra/` project. This adds a VPC, security group, and RDS PostgreSQL instance to the `IcarusDannerInfraStack`.

**Key CDK constructs:**
- `ec2.Vpc` — A new VPC (or minimal VPC) with public/private subnets for the RDS instance
- `ec2.SecurityGroup` — Allows inbound PostgreSQL (port 5432) from the developer's IP or a CIDR range
- `rds.DatabaseInstance` — PostgreSQL 16, `db.t3.micro`, single-AZ, credentials auto-generated in Secrets Manager
- `CfnOutput` — Exports the RDS endpoint, port, and secret ARN for the ETL script to use

**CDK additions to `infra-stack.ts`:**
```typescript
import * as aws_ec2 from 'aws-cdk-lib/aws-ec2';
import * as aws_rds from 'aws-cdk-lib/aws-rds';

// VPC for RDS
const vpc = new aws_ec2.Vpc(this, 'ElectionDataVpc', {
  maxAzs: 2,
  natGateways: 0,
  subnetConfiguration: [
    { name: 'public', subnetType: aws_ec2.SubnetType.PUBLIC, cidrMask: 24 },
    { name: 'isolated', subnetType: aws_ec2.SubnetType.PRIVATE_ISOLATED, cidrMask: 24 },
  ],
});

// Security group allowing PostgreSQL access
const dbSecurityGroup = new aws_ec2.SecurityGroup(this, 'ElectionDbSg', {
  vpc,
  description: 'Allow PostgreSQL access to election data RDS',
  allowAllOutbound: true,
});
dbSecurityGroup.addIngressRule(aws_ec2.Peer.anyIpv4(), aws_ec2.Port.tcp(5432), 'PostgreSQL access');

// RDS PostgreSQL instance
const electionDb = new aws_rds.DatabaseInstance(this, 'ElectionDataDb', {
  engine: aws_rds.DatabaseInstanceEngine.postgres({ version: aws_rds.PostgresEngineVersion.VER_16 }),
  instanceType: aws_ec2.InstanceType.of(aws_ec2.InstanceClass.T3, aws_ec2.InstanceSize.MICRO),
  vpc,
  vpcSubnets: { subnetType: aws_ec2.SubnetType.PUBLIC },
  securityGroups: [dbSecurityGroup],
  databaseName: 'virginia_elections',
  credentials: aws_rds.Credentials.fromGeneratedSecret('postgres'),
  multiAz: false,
  allocatedStorage: 20,
  maxAllocatedStorage: 50,
  publiclyAccessible: true,
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  deletionProtection: false,
});

// Outputs for ETL script
new cdk.CfnOutput(this, 'ElectionDbEndpoint', {
  value: electionDb.dbInstanceEndpointAddress,
  description: 'Election data RDS endpoint',
});
new cdk.CfnOutput(this, 'ElectionDbSecretArn', {
  value: electionDb.secret!.secretArn,
  description: 'Election data RDS credentials secret ARN',
});
```

**ETL connection flow:** The ETL script retrieves the RDS credentials from Secrets Manager using `boto3` (with the `icarus` profile), then connects via `psycopg2` using the endpoint from the CDK output.

### 1. Database Schema (DDL)

The schema uses `SERIAL` primary keys, foreign key constraints for referential integrity, and a polymorphic `results` table that links to either a district or a precinct.

```sql
-- Enable fuzzy text matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Elections table (one row per JSON file)
CREATE TABLE elections (
    id SERIAL PRIMARY KEY,
    record_id VARCHAR(255) NOT NULL UNIQUE,
    year SMALLINT NOT NULL,
    office VARCHAR(100) NOT NULL,
    stage VARCHAR(100) NOT NULL,
    total_votes INTEGER NOT NULL
);

-- Districts table (one row per district within an election)
CREATE TABLE districts (
    id SERIAL PRIMARY KEY,
    election_id INTEGER NOT NULL REFERENCES elections(id) ON DELETE CASCADE,
    district_name VARCHAR(100) NOT NULL,
    total_votes INTEGER NOT NULL,
    win_number NUMERIC,
    flip_number NUMERIC,
    win_gap NUMERIC
);

-- Precincts table (one row per precinct within a district)
CREATE TABLE precincts (
    id SERIAL PRIMARY KEY,
    district_id INTEGER NOT NULL REFERENCES districts(id) ON DELETE CASCADE,
    precinct_name VARCHAR(255) NOT NULL,
    total_votes INTEGER NOT NULL,
    win_number NUMERIC,
    flip_number NUMERIC,
    win_gap NUMERIC,
    county VARCHAR(100),
    precinct_code VARCHAR(50),
    precinct_label VARCHAR(255)
);

-- Results table (one row per candidate per district or precinct)
CREATE TABLE results (
    id SERIAL PRIMARY KEY,
    district_id INTEGER REFERENCES districts(id) ON DELETE CASCADE,
    precinct_id INTEGER REFERENCES precincts(id) ON DELETE CASCADE,
    candidate_name VARCHAR(255) NOT NULL,
    votes INTEGER NOT NULL,
    CONSTRAINT results_parent_check CHECK (
        (district_id IS NOT NULL AND precinct_id IS NULL) OR
        (district_id IS NULL AND precinct_id IS NOT NULL)
    )
);

-- Performance indexes
CREATE INDEX idx_elections_record_id ON elections(record_id);
CREATE INDEX idx_elections_office ON elections(office);
CREATE INDEX idx_elections_year ON elections(year);
CREATE INDEX idx_districts_district_name ON districts(district_name);
CREATE INDEX idx_districts_election_id ON districts(election_id);
CREATE INDEX idx_precincts_district_id ON precincts(district_id);
CREATE INDEX idx_results_district_id ON results(district_id);
CREATE INDEX idx_results_precinct_id ON results(precinct_id);

-- Trigram index for fuzzy precinct name search
CREATE INDEX idx_precincts_name_trgm ON precincts USING gin (precinct_name gin_trgm_ops);
```

**Design notes:**
- `votes` and `total_votes` are `INTEGER` — the source JSON uses floats (e.g., `1338.0`) but all values are whole numbers. The ETL converts with `int()`.
- `win_number`, `flip_number`, `win_gap` are `NUMERIC` (nullable) because they are optional and some files have `0` (integer, not float) as values.
- The `results_parent_check` constraint enforces that every result row belongs to exactly one parent (district XOR precinct).
- `record_id` has a `UNIQUE` constraint for idempotency — the ETL checks this before inserting.

### 2. ETL Script (`etl.py`)

A single Python script with these components:

```
etl.py
├── main()                    # CLI entry point, orchestrates the pipeline
├── get_db_connection()       # Connect to RDS PostgreSQL
├── discover_files()          # Find JSON files (local or S3)
├── read_json_file()          # Read and parse a single JSON file
├── validate_json()           # Check required top-level fields
├── parse_precinct_name()     # Parse "County_Code - Label" into components
├── process_file()            # Transform + load a single file (one transaction)
├── insert_election()         # INSERT into elections table
├── insert_district()         # INSERT into districts table
├── insert_district_results() # INSERT district-level results
├── insert_precinct()         # INSERT into precincts table
├── insert_precinct_results() # INSERT precinct-level results
└── print_summary()           # Print ingestion report
```

**CLI interface:**
```bash
python etl.py --source local --db-url postgresql://user:pass@host:5432/virginia_elections
python etl.py --source s3 --db-url postgresql://user:pass@host:5432/virginia_elections
```

**Dependencies:** `psycopg2-binary`, `boto3`, `argparse` (stdlib)

### 3. Precinct Name Parser

The parser handles 4 distinct patterns found in the data:

| Pattern | Example | County | Code | Label |
|---------|---------|--------|------|-------|
| Standard | `Accomack County_101 - Chincoteague` | `Accomack County` | `101` | `Chincoteague` |
| Arlington (no leading zeros) | `Arlington County_1 - Arlington` | `Arlington County` | `1` | `Arlington` |
| Special codes | `Alexandria City_##ab - Central Absentee Precinct` | `Alexandria City` | `##ab` | `Central Absentee Precinct` |
| Provisional (simple) | `Prince George County_Provisional` | `Prince George County` | `Provisional` | `NULL` |
| Provisional (verbose) | `Carroll County_Carroll County Provisionals` | `Carroll County` | `Provisional` | `Carroll County Provisionals` |
| Unparseable | anything else | `NULL` | `NULL` | `NULL` |

**Parsing logic (regex-based):**

```python
import re

# Pattern 1: Standard/Arlington/Special codes with " - " separator
# Matches: "County_Code - Label"
PATTERN_WITH_LABEL = re.compile(r'^(.+?)_([^_]+?) - (.+)$')

# Pattern 2: Provisional without " - " separator
# Matches: "County_Provisional" or "County_County Provisionals"
PATTERN_PROVISIONAL = re.compile(r'^(.+?)_(.*[Pp]rovisional.*)$')

def parse_precinct_name(name: str) -> dict:
    """Parse precinct name into county, code, and label components."""
    
    # Try standard pattern first (has " - " separator)
    match = PATTERN_WITH_LABEL.match(name)
    if match:
        return {
            'county': match.group(1),
            'precinct_code': match.group(2),
            'precinct_label': match.group(3),
        }
    
    # Try provisional pattern (no " - " separator)
    match = PATTERN_PROVISIONAL.match(name)
    if match:
        county = match.group(1)
        remainder = match.group(2)
        return {
            'county': county,
            'precinct_code': 'Provisional',
            'precinct_label': remainder if remainder != 'Provisional' else None,
        }
    
    # Unparseable — log warning, store full name only
    return {
        'county': None,
        'precinct_code': None,
        'precinct_label': None,
    }
```

### 4. S3 Reader

Uses `boto3` with the `icarus` AWS profile to list and download JSON objects:

```python
session = boto3.Session(profile_name='icarus')
s3 = session.client('s3')
```

The S3 reader lists all objects in `predictif-election-data`, filters for `.json` suffix, and downloads each to memory (no temp files needed — files are small).

### 5. Ingestion Reporter

Tracks counters during processing and prints a summary:

```
=== Ingestion Summary ===
Total files processed: 43
  Successful: 43
  Skipped (duplicate): 0
  Failed: 0

Rows inserted:
  elections: 43
  districts: 187
  precincts: 12,450
  results: 98,320
```

## Data Models

### JSON Source Structure (per file)

```json
{
  "record_id": "Governor_2021_General_Election",
  "year": 2021,
  "office": "Governor",
  "stage": "General_Election",
  "total_votes": 3288318.0,
  "districts": [
    {
      "district_name": "Statewide",
      "district_total_votes": 3288318.0,
      "district_win_number": 1676004.0,
      "district_flip_number": 31845.0,
      "district_win_gap": 63688.0,          // OPTIONAL — missing in some files
      "district_results": [
        { "candidate_name": "...", "votes": 1663158.0 }
      ],
      "precincts": [
        {
          "precinct_name": "Accomack County_101 - Chincoteague",
          "precinct_total_votes": 1318.0,
          "results": [
            { "candidate_name": "...", "votes": 983.0 }
          ],
          "win_number": 989.0,              // OPTIONAL
          "flip_number": 330.0,             // OPTIONAL
          "win_gap": 657.0                  // OPTIONAL
        }
      ]
    }
  ]
}
```

### Relational Model (ER Diagram)

```mermaid
erDiagram
    elections ||--o{ districts : contains
    districts ||--o{ precincts : contains
    districts ||--o{ results : "has district-level"
    precincts ||--o{ results : "has precinct-level"

    elections {
        serial id PK
        varchar record_id UK
        smallint year
        varchar office
        varchar stage
        integer total_votes
    }

    districts {
        serial id PK
        integer election_id FK
        varchar district_name
        integer total_votes
        numeric win_number
        numeric flip_number
        numeric win_gap
    }

    precincts {
        serial id PK
        integer district_id FK
        varchar precinct_name
        integer total_votes
        numeric win_number
        numeric flip_number
        numeric win_gap
        varchar county
        varchar precinct_code
        varchar precinct_label
    }

    results {
        serial id PK
        integer district_id FK
        integer precinct_id FK
        varchar candidate_name
        integer votes
    }
```

### District Name Categories

| Category | Pattern | Example Offices |
|----------|---------|-----------------|
| Statewide | `"Statewide"` | Governor, President, Attorney General, Lt. Governor, U.S. Senate |
| Local catch-all | `"District_0"` | Sheriff, Mayor, Town Council, County Board, Commonwealth's Attorney, Treasurer |
| Legislative numbered | `"District_N"` (N > 0) | House of Delegates, U.S. House, Senate of Virginia |

### Vote Type Conversion

All vote values in the JSON are floats (e.g., `1338.0`). The ETL converts to `int` before insertion. The `win_number`, `flip_number`, and `win_gap` fields use `NUMERIC` in the schema because some files contain `0` as a bare integer (not `0.0`), and these fields are optional.


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: JSON validation accepts iff all required fields present

*For any* dictionary with an arbitrary subset of the keys `{record_id, year, office, stage, total_votes, districts}`, the validation function SHALL return valid if and only if all six keys are present.

**Validates: Requirements 3.4, 3.5**

### Property 2: Precinct name parsing round-trip

*For any* randomly generated precinct name following one of the recognized patterns (standard `"County_Code - Label"`, special codes `"County_##xx - Label"`, or provisional `"County_Provisional"`), parsing the name and reconstructing it from the parsed components SHALL produce the original name.

**Validates: Requirements 5.1, 5.2, 5.3**

### Property 3: Optional fields stored correctly

*For any* district or precinct dictionary where optional fields (`win_gap`, `win_number`, `flip_number`) are randomly present or absent, the ETL's field extraction SHALL return the field's value when present and `None` when absent.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

### Property 4: All candidate results stored

*For any* list of candidate results with length N (1 ≤ N ≤ 20), inserting them into the results table SHALL produce exactly N rows, each with the correct candidate name and vote count.

**Validates: Requirements 7.1, 7.2**

### Property 5: Float-to-int vote preservation

*For any* non-negative integer V, converting the float representation `float(V)` to `int` SHALL produce the original value V.

**Validates: Requirements 7.3**

### Property 6: Idempotent ingestion

*For any* valid election JSON, processing it twice against the same database SHALL result in exactly one row in the `elections` table with that `record_id`, and the second processing SHALL be skipped.

**Validates: Requirements 8.1, 8.2**

### Property 7: Transaction atomicity on failure

*For any* valid election JSON where an error is injected at a random point during ingestion, after the transaction rolls back, there SHALL be zero rows in any table (`elections`, `districts`, `precincts`, `results`) associated with that file's `record_id`.

**Validates: Requirements 8.3, 8.4**

## Error Handling

### Connection Errors
- If the database connection fails, the ETL logs the error (including any AWS/psycopg2 error code) and exits with a non-zero status code. No partial processing occurs.

### File-Level Errors
- Each JSON file is processed in its own transaction. If any step fails (validation, parsing, insertion), the transaction is rolled back and the file is recorded as failed.
- The ETL continues to the next file — one bad file does not stop the entire pipeline.

### Validation Errors
- Missing required top-level fields: file is skipped with a warning log identifying the file and missing fields.
- Unparseable precinct names: the full name is stored in `precinct_name`, parsed fields are set to NULL, and a warning is logged. The file is NOT skipped — only the parsing is degraded.
- Unrecognized district names: stored as-is with a warning log. The file is NOT skipped.

### Duplicate Handling
- Before inserting, the ETL queries for an existing `record_id`. If found, the file is skipped with an info-level log. This is not an error — it's expected behavior for re-runs.

### Data Type Errors
- If a vote value cannot be converted to int (e.g., `NaN`, non-numeric), the entire file's transaction is rolled back and the file is recorded as failed.

## Testing Strategy

### Unit Tests (example-based)
- **CLI argument parsing**: Verify `--source local` and `--source s3` are accepted; invalid values rejected.
- **Schema validation**: Verify all tables, columns, constraints, and indexes exist after DDL execution.
- **District name recognition**: Verify "Statewide", "District_0", "District_100" are recognized; random strings trigger warnings.
- **Arlington format**: Verify `"Arlington County_1 - Arlington"` parses correctly (no leading zeros).
- **Summary report format**: Verify the report contains correct counts after a known ingestion run.
- **S3 integration**: Mock boto3 client, verify list/download calls.
- **Error recovery**: Process a batch with one bad file, verify others succeed.

### Property-Based Tests (using `hypothesis` library)
- **Minimum 100 iterations per property** (Hypothesis default is 100+).
- Each test tagged with: `Feature: election-data-migration, Property N: <title>`
- Properties 1–7 from the Correctness Properties section above.
- Generators:
  - Random subsets of required JSON fields (Property 1)
  - Random county names, precinct codes, labels, special codes (Property 2)
  - Random district/precinct dicts with optional fields toggled (Property 3)
  - Random candidate lists of varying length (Property 4)
  - Random non-negative integers for vote conversion (Property 5)
  - Random valid election JSON for idempotency (Property 6)
  - Random valid election JSON with injected failures (Property 7)

### Integration Tests
- **End-to-end ingestion**: Run the ETL against a test PostgreSQL instance with a subset of real JSON files, verify row counts match expected values.
- **S3 mode**: Run against a mocked or real S3 bucket, verify all files are discovered and processed.
- **Fuzzy search**: After ingestion, verify `SELECT * FROM precincts WHERE precinct_name % 'Chincoateague'` returns the Chincoteague precinct (testing pg_trgm).
