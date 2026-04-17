# Implementation Plan: Election Data Migration

## Overview

Migrate 43 Virginia election JSON files into a normalized PostgreSQL schema on AWS RDS. The RDS instance is provisioned via CDK (TypeScript) within the existing `icarus-cdk` project. The implementation produces three runnable artifacts: CDK infrastructure code for RDS provisioning, a DDL script (`schema.sql`), and a Python ETL script (`etl.py`), plus property-based and integration tests. Local file reading is the default source; S3 is an optional flag.

## Tasks

- [x] 1. Add RDS PostgreSQL infrastructure to CDK stack
  - [x] 1.1 Add VPC, security group, and RDS PostgreSQL instance to `icarus-cdk/infra/lib/infra-stack.ts`
    - Create a VPC with public and isolated subnets (no NAT gateways to minimize cost)
    - Create a security group allowing inbound PostgreSQL (port 5432)
    - Create an `rds.DatabaseInstance` with PostgreSQL 16, `db.t3.micro`, single-AZ, `virginia_elections` database name
    - Store credentials in Secrets Manager via `Credentials.fromGeneratedSecret('postgres')`
    - Make the instance publicly accessible for ETL script access
    - Set `removalPolicy: DESTROY` and `deletionProtection: false` for dev environment
    - Add `CfnOutput` for the RDS endpoint and Secrets Manager secret ARN
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 2. Create the PostgreSQL schema DDL script
  - [x] 2.1 Create `schema.sql` with all table definitions, constraints, and indexes
    - Enable `pg_trgm` extension
    - Create `elections` table with `record_id UNIQUE` constraint
    - Create `districts` table with FK to `elections` and nullable `win_number`, `flip_number`, `win_gap`
    - Create `precincts` table with FK to `districts`, parsed columns (`county`, `precinct_code`, `precinct_label`), and nullable optional fields
    - Create `results` table with polymorphic FK (`district_id` XOR `precinct_id`) and `results_parent_check` CHECK constraint
    - Add all performance indexes including `gin_trgm_ops` trigram index on `precinct_name`
    - _Requirements: 1.7, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

- [x] 3. Implement core ETL components
  - [x] 3.1 Create `etl.py` with CLI argument parsing, DB connection, and main orchestration loop
    - Use `argparse` for `--source` (local/s3) and `--db-url` arguments
    - Implement `get_db_connection()` using `psycopg2`
    - Implement `main()` that discovers files, processes each, and prints summary
    - Wire up logging (stdout, with WARNING/INFO levels)
    - _Requirements: 1.8, 3.1, 8.3, 8.4, 9.1_

  - [x] 3.2 Implement local file discovery and S3 file reader
    - `discover_files()`: recursively find `.json` files under `election-data/` for local mode
    - S3 mode: use `boto3` with `profile_name='icarus'` to list and download from `predictif-election-data`
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 3.3 Implement JSON validation function
    - `validate_json()`: check for required top-level fields (`record_id`, `year`, `office`, `stage`, `total_votes`, `districts`)
    - Return list of missing fields; log warning and skip file if any missing
    - _Requirements: 3.4, 3.5_

  - [ ]* 3.4 Write property test: JSON validation (Property 1)
    - **Property 1: JSON validation accepts iff all required fields present**
    - Generate random dicts with arbitrary subsets of the 6 required keys
    - Assert validation passes iff all 6 keys present
    - **Validates: Requirements 3.4, 3.5**

  - [x] 3.5 Implement precinct name parser
    - `parse_precinct_name()`: regex-based parser handling standard, Arlington, special code, provisional, and unparseable patterns
    - Return dict with `county`, `precinct_code`, `precinct_label` (NULLs for unparseable)
    - Log warning for unparseable names
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 3.6 Write property test: Precinct name parsing round-trip (Property 2)
    - **Property 2: Precinct name parsing round-trip**
    - Generate random precinct names in recognized patterns (standard, special codes, provisional)
    - Parse and reconstruct; assert reconstructed name equals original
    - **Validates: Requirements 5.1, 5.2, 5.3**

- [x] 4. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement data insertion layer
  - [x] 5.1 Implement election, district, and precinct insertion functions
    - `insert_election()`: INSERT into `elections`, return new `id`
    - `insert_district()`: INSERT into `districts` with optional fields handled, return new `id`
    - `insert_district_results()`: INSERT one row per candidate into `results` with `district_id`
    - `insert_precinct()`: INSERT into `precincts` with parsed name components and optional fields, return new `id`
    - `insert_precinct_results()`: INSERT one row per candidate into `results` with `precinct_id`
    - Convert float votes to `int()` before insertion
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3_

  - [ ]* 5.2 Write property test: Optional fields stored correctly (Property 3)
    - **Property 3: Optional fields stored correctly**
    - Generate random district/precinct dicts with optional fields randomly present or absent
    - Assert extraction returns value when present, `None` when absent
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

  - [ ]* 5.3 Write property test: All candidate results stored (Property 4)
    - **Property 4: All candidate results stored**
    - Generate random candidate lists (length 1–20)
    - Assert insertion produces exactly N rows with correct names and vote counts
    - **Validates: Requirements 7.1, 7.2**

  - [ ]* 5.4 Write property test: Float-to-int vote preservation (Property 5)
    - **Property 5: Float-to-int vote preservation**
    - Generate random non-negative integers, convert `float(V)` → `int`
    - Assert result equals original V
    - **Validates: Requirements 7.3**

- [x] 6. Implement transaction management and idempotency
  - [x] 6.1 Implement `process_file()` with per-file transaction wrapping and idempotency check
    - Query `elections` for existing `record_id` before inserting — skip if found
    - Wrap all inserts (election → districts + district results → precincts + precinct results) in a single transaction
    - Commit on success, rollback on any error
    - Log skipped/failed files and continue to next file
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [ ]* 6.2 Write property test: Idempotent ingestion (Property 6)
    - **Property 6: Idempotent ingestion**
    - Generate valid election JSON, process twice against test DB
    - Assert exactly one `elections` row with that `record_id`; second run skipped
    - **Validates: Requirements 8.1, 8.2**

  - [ ]* 6.3 Write property test: Transaction atomicity on failure (Property 7)
    - **Property 7: Transaction atomicity on failure**
    - Generate valid election JSON, inject error at random point during ingestion
    - Assert zero rows in all tables for that `record_id` after rollback
    - **Validates: Requirements 8.3, 8.4**

- [x] 7. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement ingestion reporter and wire everything together
  - [x] 8.1 Implement `print_summary()` with counters for files processed, succeeded, skipped, failed, and per-table row counts
    - Track counters during processing loop in `main()`
    - Print failed file names with error messages if any
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 8.2 Wire all components together in `main()` and verify end-to-end local mode works
    - Ensure `main()` calls discover → validate → process → summary in order
    - Add `requirements.txt` with `psycopg2-binary`, `boto3`, `hypothesis` (dev)
    - _Requirements: 3.1, 3.2, 8.3, 8.4, 9.1_

  - [ ]* 8.3 Write integration tests for end-to-end ingestion
    - Run ETL against a test PostgreSQL instance with a subset of real JSON files
    - Verify row counts match expected values across all 4 tables
    - Verify fuzzy search with `pg_trgm` returns expected precinct for a misspelled query
    - _Requirements: 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 7.1, 7.2_

- [x] 9. Add README with usage instructions
  - Document prerequisites (Python 3.10+, PostgreSQL, AWS CLI with `icarus` profile)
  - Document how to deploy the CDK stack (`cdk deploy`) to provision the RDS instance
  - Document how to retrieve RDS credentials from Secrets Manager
  - Document how to run `schema.sql` against the RDS instance
  - Document how to run `etl.py` in local and S3 modes
  - Document how to run tests (`pytest`)
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 3.1_

- [x] 10. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate the 7 correctness properties from the design document using `hypothesis`
- The ETL reads from local `election-data/` by default; S3 is toggled via `--source s3`
- The user will provision the RDS instance and run the scripts themselves — tasks produce runnable artifacts
