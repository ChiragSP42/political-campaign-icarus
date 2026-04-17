#!/usr/bin/env python3
"""ETL script to migrate Virginia election JSON files into PostgreSQL.

Uses batch inserts (execute_values) for 10-20x speedup over individual INSERTs.
Writes two log files and uploads them to S3 on completion.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime

import psycopg2
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# ANSI color codes
# ---------------------------------------------------------------------------
GREEN = "\x1b[32m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"
MAGENTA = "\x1b[35m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = {"record_id", "year", "office", "stage", "total_votes", "districts"}
PATTERN_WITH_LABEL = re.compile(r"^(.+?)_([^_]+?) - (.+)$")
PATTERN_PROVISIONAL = re.compile(r"^(.+?)_(.*[Pp]rovisional.*)$")
S3_LOG_BUCKET = "predictif-election-data"
S3_LOG_PREFIX = "etl-logs/"


def safe_int(value, default=0):
    """Convert a value to int, handling NaN and None gracefully."""
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return int(value)

# ---------------------------------------------------------------------------
# Log files
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
full_log = None
errors_log = None
full_log_path = None
errors_log_path = None

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def log(msg, level="INFO"):
    print(msg, flush=True)
    # ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # plain = strip_ansi(msg)
    # if full_log:
    #     full_log.write(f"[{ts}] {plain}\n")
    #     full_log.flush()
    # if level in ("WARN", "ERROR") and errors_log:
    #     errors_log.write(f"[{ts}] [{level}] {plain}\n")
    #     errors_log.flush()


def log_end(msg, level="INFO"):
    print(msg, end="", flush=True)
    # ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # plain = strip_ansi(msg)
    # if full_log:
    #     full_log.write(f"[{ts}] {plain}")
    #     full_log.flush()


def log_cont(msg, level="INFO"):
    print(msg, flush=True)
    # plain = strip_ansi(msg)
    # if full_log:
    #     full_log.write(f"{plain}\n")
    #     full_log.flush()
    # if level in ("WARN", "ERROR") and errors_log:
    #     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     errors_log.write(f"[{ts}] [{level}] {plain}\n")
    #     errors_log.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Election data ETL (batch mode)")
    parser.add_argument("--source", choices=["local", "s3"], default="s3",
                        help="Data source: local files or S3 bucket (default: s3)")
    parser.add_argument("--db-url", required=True,
                        help="PostgreSQL connection URL")
    parser.add_argument("--upload-logs", action="store_true", default=False,
                        help="Upload log files to S3 on completion")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------
def get_db_connection(db_url):
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    return conn


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def discover_local_files(base_dir="election-data"):
    json_files = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(".json"):
                json_files.append(os.path.join(root, fname))
    json_files.sort()
    return json_files


def discover_s3_files(bucket=S3_LOG_BUCKET):
    import boto3
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json") and not key.startswith(S3_LOG_PREFIX):
                log(f"  {CYAN}Downloading{RESET} {key}")
                response = s3.get_object(Bucket=bucket, Key=key)
                body = response["Body"].read()
                data = json.loads(body)
                files.append((key, data))
    files.sort(key=lambda x: x[0])
    return files


def read_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_json(data, filepath):
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        log(f"  {YELLOW}⚠ WARN{RESET} Missing required fields in {filepath}: {sorted(missing)}", level="WARN")
        return False, sorted(missing)
    return True, []


# ---------------------------------------------------------------------------
# Precinct name parser
# ---------------------------------------------------------------------------
def parse_precinct_name(name):
    match = PATTERN_WITH_LABEL.match(name)
    if match:
        return {"county": match.group(1), "precinct_code": match.group(2), "precinct_label": match.group(3)}

    match = PATTERN_PROVISIONAL.match(name)
    if match:
        county = match.group(1)
        remainder = match.group(2)
        return {"county": county, "precinct_code": "Provisional",
                "precinct_label": remainder if remainder != "Provisional" else None}

    log(f"    {YELLOW}⚠ Unparseable precinct name:{RESET} {name}", level="WARN")
    return {"county": None, "precinct_code": None, "precinct_label": None}


# ---------------------------------------------------------------------------
# Batch insertion functions
# ---------------------------------------------------------------------------
def insert_election(cursor, data):
    cursor.execute(
        "INSERT INTO elections (record_id, year, office, stage, total_votes) VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (data["record_id"], int(data["year"]), data["office"], data["stage"], safe_int(data["total_votes"])))
    return cursor.fetchone()[0]


def insert_district(cursor, election_id, district):
    cursor.execute(
        "INSERT INTO districts (election_id, district_name, total_votes, win_number, flip_number, win_gap) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        (election_id, district["district_name"], safe_int(district["district_total_votes"]),
         district.get("district_win_number"), district.get("district_flip_number"), district.get("district_win_gap")))
    return cursor.fetchone()[0]


def batch_insert_results(cursor, rows):
    """Batch insert results using execute_values. Each row is (district_id, precinct_id, candidate_name, votes)."""
    if not rows:
        return
    execute_values(
        cursor,
        "INSERT INTO results (district_id, precinct_id, candidate_name, votes) VALUES %s",
        rows,
        page_size=500,
    )


def batch_insert_precincts_and_results(cursor, district_id, precincts_data):
    """Batch insert all precincts for a district, then batch insert all their results.

    Returns (num_precincts, num_results).
    """
    if not precincts_data:
        return 0, 0

    # Build precinct rows
    precinct_rows = []
    parsed_list = []
    for p in precincts_data:
        parsed = parse_precinct_name(p["precinct_name"])
        parsed_list.append(parsed)
        precinct_rows.append((
            district_id, p["precinct_name"], safe_int(p["precinct_total_votes"]),
            p.get("win_number"), p.get("flip_number"), p.get("win_gap"),
            parsed["county"], parsed["precinct_code"], parsed["precinct_label"],
        ))

    # Batch insert precincts and get their IDs
    precinct_ids = execute_values(
        cursor,
        """INSERT INTO precincts (district_id, precinct_name, total_votes,
                                  win_number, flip_number, win_gap,
                                  county, precinct_code, precinct_label)
           VALUES %s RETURNING id""",
        precinct_rows,
        fetch=True,
        page_size=500,
    )

    # Build results rows using returned precinct IDs
    result_rows = []
    for i, p in enumerate(precincts_data):
        pid = precinct_ids[i][0]
        for r in p.get("results", []):
            result_rows.append((None, pid, r["candidate_name"], safe_int(r["votes"])))

    batch_insert_results(cursor, result_rows)
    return len(precincts_data), len(result_rows)


# ---------------------------------------------------------------------------
# Transaction management + idempotency
# ---------------------------------------------------------------------------
def process_file(conn, data, filepath, file_num, total_files):
    record_id = data["record_id"]
    num_districts = len(data.get("districts", []))
    num_precincts = sum(len(d.get("precincts", [])) for d in data.get("districts", []))

    log(f"\n{BOLD}[{file_num}/{total_files}]{RESET} {CYAN}Processing:{RESET} {record_id}")
    log(f"  File: {filepath}")
    log(f"  Districts: {num_districts}, Precincts: {num_precincts}")

    cursor = conn.cursor()

    cursor.execute("SELECT 1 FROM elections WHERE record_id = %s", (record_id,))
    if cursor.fetchone():
        log(f"  {YELLOW}⏭ SKIPPED{RESET} — record_id '{record_id}' already exists")
        cursor.close()
        return "skipped"

    counters = {"elections": 0, "districts": 0, "precincts": 0, "results": 0}
    start = time.time()

    try:
        log_end(f"  {MAGENTA}→ Inserting election...{RESET}")
        election_id = insert_election(cursor, data)
        counters["elections"] += 1
        log_cont(f" {GREEN}✓{RESET}")

        for i, district in enumerate(data["districts"], 1):
            d_name = district["district_name"]
            d_precincts = district.get("precincts", [])
            d_results = district.get("district_results", [])

            log(f"  {MAGENTA}→ District {i}/{num_districts}: {d_name}{RESET} ({len(d_precincts)} precincts, {len(d_results)} candidates)")

            district_id = insert_district(cursor, election_id, district)
            counters["districts"] += 1

            # Batch insert district-level results
            dist_result_rows = [(district_id, None, r["candidate_name"], safe_int(r["votes"])) for r in d_results]
            batch_insert_results(cursor, dist_result_rows)
            counters["results"] += len(dist_result_rows)

            # Batch insert precincts + their results
            np, nr = batch_insert_precincts_and_results(cursor, district_id, d_precincts)
            counters["precincts"] += np
            counters["results"] += nr

            log(f"    {GREEN}✓ {np} precincts, {nr + len(dist_result_rows)} results{RESET}")

        log_end(f"  {MAGENTA}→ Committing transaction...{RESET}")
        conn.commit()
        elapsed = time.time() - start
        log_cont(f" {GREEN}✓{RESET}")
        log(f"  {GREEN}✓ SUCCESS{RESET} in {elapsed:.1f}s — {counters['precincts']} precincts, {counters['results']} results")
        return counters

    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# Summary reporter
# ---------------------------------------------------------------------------
def print_summary(results):
    total = len(results)
    succeeded = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")

    totals = {"elections": 0, "districts": 0, "precincts": 0, "results": 0}
    for r in results:
        if r["status"] == "success" and r["counters"]:
            for table, count in r["counters"].items():
                totals[table] += count

    log(f"\n{BOLD}{'='*50}{RESET}")
    log(f"{BOLD}  Ingestion Summary{RESET}")
    log(f"{BOLD}{'='*50}{RESET}")
    log(f"  Total files processed: {BOLD}{total}{RESET}")
    log(f"  {GREEN}Successful: {succeeded}{RESET}")
    log(f"  {YELLOW}Skipped (duplicate): {skipped}{RESET}")
    log(f"  {RED}Failed: {failed}{RESET}")
    log("")
    log(f"  {BOLD}Rows inserted:{RESET}")
    for table, count in totals.items():
        log(f"    {table}: {CYAN}{count:,}{RESET}")

    failed_files = [r for r in results if r["status"] == "failed"]
    if failed_files:
        log("")
        log(f"  {RED}{BOLD}Failed files:{RESET}")
        for r in failed_files:
            log(f"    {RED}✗{RESET} {r['filepath']}: {r['error']}", level="ERROR")
    log(f"{BOLD}{'='*50}{RESET}")


# ---------------------------------------------------------------------------
# S3 log upload
# ---------------------------------------------------------------------------
def upload_logs_to_s3():
    """Upload both log files to S3 for later retrieval."""
    try:
        import boto3
        s3 = boto3.client("s3")
        for local_path in [full_log_path, errors_log_path]:
            if local_path and os.path.exists(local_path):
                key = S3_LOG_PREFIX + os.path.basename(local_path)
                s3.upload_file(local_path, S3_LOG_BUCKET, key)
                log(f"  {GREEN}✓ Uploaded{RESET} s3://{S3_LOG_BUCKET}/{key}")
    except Exception as e:
        print(f"  {RED}✗ Failed to upload logs to S3: {e}{RESET}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    global full_log, errors_log, full_log_path, errors_log_path

    args = parse_args(argv)

    # Create logs directory
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_log_path = os.path.join(LOG_DIR, f"etl_full_{ts}.log")
    errors_log_path = os.path.join(LOG_DIR, f"etl_errors_{ts}.log")

    # full_log = open(full_log_path, "w", encoding="utf-8")
    # errors_log = open(errors_log_path, "w", encoding="utf-8")

    log(f"{CYAN}Log files:{RESET}")
    log(f"  Full log:   {full_log_path}")
    log(f"  Errors log: {errors_log_path}")
    log(f"  Mode: {BOLD}BATCH INSERTS{RESET} (execute_values)")

    # Connect
    log_end(f"{CYAN}Connecting to database...{RESET}")
    try:
        conn = get_db_connection(args.db_url)
        log_cont(f" {GREEN}✓ Connected{RESET}")
    except Exception as e:
        log_cont(f" {RED}✗ FAILED: {e}{RESET}", level="ERROR")
        # full_log.close()
        # errors_log.close()
        sys.exit(1)

    results = []

    try:
        if args.source == "s3":
            log(f"{CYAN}Discovering files from S3 ({S3_LOG_BUCKET})...{RESET}")
            s3_files = discover_s3_files()
            total_files = len(s3_files)
            log(f"{GREEN}Found {total_files} JSON files in S3{RESET}")

            for idx, (key, data) in enumerate(s3_files, 1):
                valid, missing = validate_json(data, key)
                if not valid:
                    results.append({"filepath": key, "status": "failed", "counters": None, "error": f"Missing fields: {missing}"})
                    continue
                try:
                    outcome = process_file(conn, data, key, idx, total_files)
                    if outcome == "skipped":
                        results.append({"filepath": key, "status": "skipped", "counters": None, "error": None})
                    else:
                        results.append({"filepath": key, "status": "success", "counters": outcome, "error": None})
                except Exception as e:
                    log(f"  {RED}✗ FAILED: {e}{RESET}", level="ERROR")
                    results.append({"filepath": key, "status": "failed", "counters": None, "error": str(e)})
        else:
            log_end(f"{CYAN}Discovering local files...{RESET}")
            filepaths = discover_local_files()
            total_files = len(filepaths)
            log_cont(f" {GREEN}Found {total_files} JSON files{RESET}")

            for idx, filepath in enumerate(filepaths, 1):
                try:
                    data = read_json_file(filepath)
                except Exception as e:
                    log(f"  {RED}✗ Failed to read {filepath}: {e}{RESET}", level="ERROR")
                    results.append({"filepath": filepath, "status": "failed", "counters": None, "error": str(e)})
                    continue

                valid, missing = validate_json(data, filepath)
                if not valid:
                    results.append({"filepath": filepath, "status": "failed", "counters": None, "error": f"Missing fields: {missing}"})
                    continue

                try:
                    outcome = process_file(conn, data, filepath, idx, total_files)
                    if outcome == "skipped":
                        results.append({"filepath": filepath, "status": "skipped", "counters": None, "error": None})
                    else:
                        results.append({"filepath": filepath, "status": "success", "counters": outcome, "error": None})
                except Exception as e:
                    log(f"  {RED}✗ FAILED: {e}{RESET}", level="ERROR")
                    results.append({"filepath": filepath, "status": "failed", "counters": None, "error": str(e)})

        print_summary(results)

    finally:
        conn.close()
        log(f"\n{CYAN}Database connection closed{RESET}")

        # Upload logs to S3 BEFORE closing log files
        if args.upload_logs:
            log(f"{CYAN}Uploading logs to S3...{RESET}")
            upload_logs_to_s3()

        # full_log.close()
        # errors_log.close()

        print(f"\n{GREEN}Done.{RESET}", flush=True)


if __name__ == "__main__":
    main()
