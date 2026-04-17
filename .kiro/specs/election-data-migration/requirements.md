# Requirements Document

## Introduction

This feature migrates Virginia election data from local JSON files (or an S3 bucket) into a PostgreSQL database hosted on AWS RDS. The dataset consists of 43 deeply nested JSON files covering elections from 2020–2025 across 14 office types and 4 election stages. The data contains known inconsistencies in district naming, optional fields, and precinct code formatting that the ingestion pipeline must handle gracefully. The scope explicitly excludes any chatbot or tool-calling functionality — the user will build that layer separately.

## Glossary

- **RDS_Instance**: An AWS RDS PostgreSQL database instance (not Aurora Serverless) provisioned via CDK within the existing `icarus-cdk` project to store the election data.
- **ETL_Script**: A one-time Python data ingestion script that reads JSON election files, transforms the nested structures, and loads them into the relational schema.
- **Election_Schema**: The set of PostgreSQL tables (elections, districts, precincts, results) that flatten the nested JSON hierarchy into a relational model.
- **Source_JSON**: One of the 43 JSON files following the hierarchy: Election → Districts → Precincts → Results. Located either locally in `election-data/` or in the S3 bucket `predictif-election-data`.
- **Precinct_Name**: A composite string following the pattern `"County_Code - Label"` (e.g., `"Accomack County_101 - Chincoteague"`), with known variations in code formatting across counties.
- **District_Name**: A string identifying the electoral district. Uses `"Statewide"` for statewide races, `"District_0"` for local catch-all races, and numbered formats like `"District_100"` for legislative districts.
- **Win_Gap**: An optional numeric field present at both district and precinct levels in some files but absent in others, representing the vote margin between the winner and the threshold.

## Requirements

### Requirement 1: RDS PostgreSQL Instance Provisioning via CDK

**User Story:** As a developer, I want a PostgreSQL database on AWS RDS provisioned via CDK, so that I have a reliable, infrastructure-as-code managed relational store for the election data that fits into the existing `icarus-cdk` project.

#### Acceptance Criteria

1. THE RDS_Instance SHALL be provisioned as a CDK construct within the existing `icarus-cdk/infra/lib/infra-stack.ts` stack (or a dedicated construct file imported by it).
2. THE RDS_Instance SHALL be a PostgreSQL engine instance (version 14 or higher) on AWS RDS using a `db.t3.micro` or `db.t4g.micro` instance class appropriate for the dataset size (43 files, under 100MB total).
3. THE CDK stack SHALL create a VPC (or use a default/existing VPC) with the necessary subnets and security groups to host the RDS instance.
4. THE RDS_Instance SHALL be configured with a dedicated database named `virginia_elections`.
5. THE RDS_Instance SHALL store its credentials in AWS Secrets Manager (CDK default behavior for `DatabaseInstance`).
6. THE CDK stack SHALL output the RDS endpoint, port, and Secrets Manager secret ARN as CloudFormation outputs so the ETL script can connect.
7. THE RDS_Instance SHALL have the `pg_trgm` extension enabled to support fuzzy text matching on precinct names (enabled via the schema DDL script after provisioning).
8. IF the RDS_Instance fails to provision, THEN THE ETL_Script SHALL log a descriptive error message including the AWS error code and exit gracefully.

### Requirement 2: Relational Schema Design

**User Story:** As a developer, I want a flattened relational schema that maps the nested JSON hierarchy into normalized tables, so that I can run efficient SQL queries across elections, districts, precincts, and candidate results.

#### Acceptance Criteria

1. THE Election_Schema SHALL contain an `elections` table with columns: `id` (primary key), `record_id` (unique), `year`, `office`, `stage`, and `total_votes`.
2. THE Election_Schema SHALL contain a `districts` table with columns: `id` (primary key), `election_id` (foreign key to elections), `district_name`, `total_votes`, `win_number`, `flip_number`, and `win_gap` (nullable).
3. THE Election_Schema SHALL contain a `precincts` table with columns: `id` (primary key), `district_id` (foreign key to districts), `precinct_name`, `total_votes`, `win_number`, `flip_number`, `win_gap` (nullable), `county` (parsed), `precinct_code` (parsed), and `precinct_label` (parsed).
4. THE Election_Schema SHALL contain a `results` table with columns: `id` (primary key), `candidate_name`, `votes`, and either `district_id` or `precinct_id` (foreign key) to associate results with their parent level.
5. THE Election_Schema SHALL enforce referential integrity via foreign key constraints between elections → districts → precincts and between districts/precincts → results.
6. THE Election_Schema SHALL create a trigram index on the `precincts.precinct_name` column using the `pg_trgm` extension.
7. THE Election_Schema SHALL create indexes on `elections.record_id`, `elections.office`, `elections.year`, and `districts.district_name` for query performance.

### Requirement 3: JSON Data Source Reading

**User Story:** As a developer, I want the ingestion script to read election JSON files from either local disk or S3, so that I can ingest data regardless of where it is stored.

#### Acceptance Criteria

1. THE ETL_Script SHALL accept a command-line argument or configuration flag to select between local file mode (reading from `election-data/` directory) and S3 mode (reading from the `predictif-election-data` bucket using the `icarus` AWS profile).
2. WHEN operating in local mode, THE ETL_Script SHALL recursively discover all `.json` files under the `election-data/` directory.
3. WHEN operating in S3 mode, THE ETL_Script SHALL list and download all `.json` objects from the `predictif-election-data` bucket.
4. THE ETL_Script SHALL parse each Source_JSON file and validate that it contains the required top-level fields: `record_id`, `year`, `office`, `stage`, `total_votes`, and `districts`.
5. IF a Source_JSON file is missing required top-level fields, THEN THE ETL_Script SHALL log a warning identifying the file and the missing fields, skip that file, and continue processing remaining files.

### Requirement 4: District Name Normalization

**User Story:** As a developer, I want the ingestion script to handle the three distinct district naming conventions consistently, so that queries can reliably filter by district type.

#### Acceptance Criteria

1. WHEN the District_Name value is `"Statewide"`, THE ETL_Script SHALL store it as-is, indicating a statewide race (Governor, President, Attorney General, Lieutenant Governor, U.S. Senate).
2. WHEN the District_Name value is `"District_0"`, THE ETL_Script SHALL store it as-is, indicating a local catch-all race (Sheriff, Mayor, Town Council, County Board Member, Commonwealth's Attorney, Treasurer).
3. WHEN the District_Name value matches the pattern `"District_N"` where N is a positive integer, THE ETL_Script SHALL store it as-is, indicating a legislative district (House of Delegates, U.S. House, Senate of Virginia).
4. IF a District_Name value does not match any of the three recognized patterns, THEN THE ETL_Script SHALL log a warning with the file name and the unrecognized district name, and store the value as-is.

### Requirement 5: Precinct Name Parsing

**User Story:** As a developer, I want precinct names parsed into structured components (county, code, label), so that I can query and group results by county or precinct code independently.

#### Acceptance Criteria

1. WHEN a Precinct_Name follows the standard pattern `"County_Code - Label"`, THE ETL_Script SHALL parse it into three components: `county` (text before the underscore-delimited code), `precinct_code` (the numeric or alphanumeric code), and `precinct_label` (text after the ` - ` separator).
2. WHEN a Precinct_Name contains a special precinct code (`##ab`, `##ev`, `##pe`, or `Provisional`), THE ETL_Script SHALL parse the county component and store the special code in `precinct_code` with the corresponding label in `precinct_label`.
3. WHEN a Precinct_Name uses the Arlington County format (e.g., `"Arlington County_1 - Arlington"` with no leading zeros), THE ETL_Script SHALL parse it using the same logic as standard precincts without requiring leading zeros.
4. IF a Precinct_Name does not match any recognized pattern, THEN THE ETL_Script SHALL store the full name in `precinct_name`, set `county`, `precinct_code`, and `precinct_label` to NULL, and log a warning.

### Requirement 6: Optional Field Handling

**User Story:** As a developer, I want the ingestion script to handle missing optional fields gracefully, so that files with and without `win_gap` data are both ingested without errors.

#### Acceptance Criteria

1. WHEN a district object in the Source_JSON contains a `district_win_gap` field, THE ETL_Script SHALL store its value in the `win_gap` column of the `districts` table.
2. WHEN a district object in the Source_JSON does not contain a `district_win_gap` field, THE ETL_Script SHALL store NULL in the `win_gap` column of the `districts` table.
3. WHEN a precinct object in the Source_JSON contains a `win_gap` field, THE ETL_Script SHALL store its value in the `win_gap` column of the `precincts` table.
4. WHEN a precinct object in the Source_JSON does not contain a `win_gap` field, THE ETL_Script SHALL store NULL in the `win_gap` column of the `precincts` table.
5. WHEN a precinct object in the Source_JSON does not contain `win_number` or `flip_number` fields, THE ETL_Script SHALL store NULL in the corresponding columns of the `precincts` table.

### Requirement 7: Multi-Candidate and Multi-Seat Race Support

**User Story:** As a developer, I want the schema and ingestion script to handle races with varying numbers of candidates (from 2 to 10+), so that multi-seat races like Town Council are fully captured.

#### Acceptance Criteria

1. THE Election_Schema SHALL support an arbitrary number of result rows per district or precinct, with no hard-coded candidate limit.
2. WHEN a district or precinct contains multiple candidate results, THE ETL_Script SHALL insert one result row per candidate, each linked to the parent district or precinct.
3. THE ETL_Script SHALL preserve the candidate vote count as a numeric value, handling the float format present in the Source_JSON (e.g., `1338.0`) by storing it as an integer or numeric type.

### Requirement 8: Data Integrity and Idempotency

**User Story:** As a developer, I want the ingestion script to be safely re-runnable, so that I can re-execute it without creating duplicate records.

#### Acceptance Criteria

1. THE ETL_Script SHALL use the `record_id` field as a unique constraint on the `elections` table to prevent duplicate election records.
2. WHEN the ETL_Script encounters a Source_JSON whose `record_id` already exists in the database, THE ETL_Script SHALL skip that file and log an informational message.
3. THE ETL_Script SHALL wrap the ingestion of each Source_JSON file in a database transaction, committing on success and rolling back on failure for that file.
4. IF a transaction fails for a single file, THEN THE ETL_Script SHALL log the error with the file name and continue processing remaining files.

### Requirement 9: Ingestion Reporting

**User Story:** As a developer, I want a summary report after ingestion completes, so that I can verify the data was loaded correctly.

#### Acceptance Criteria

1. WHEN the ETL_Script completes processing all files, THE ETL_Script SHALL print a summary report containing: total files processed, files successfully ingested, files skipped (duplicates), and files failed.
2. THE ETL_Script SHALL print the total count of rows inserted into each table (elections, districts, precincts, results).
3. IF any files failed during ingestion, THEN THE ETL_Script SHALL list the failed file names and their error messages in the summary report.
