# Election Data Migration

One-time ETL pipeline to migrate 43 Virginia election JSON files into a PostgreSQL database on AWS RDS.

## Prerequisites

- Python 3.10+
- AWS CLI configured with the `icarus` profile (`aws configure --profile icarus`)
- CDK CLI (`npm install -g aws-cdk`)
- PostgreSQL client (`psql`) for running the schema script

## 1. Deploy the RDS Instance

The CDK stack provisions a VPC, security group, and RDS PostgreSQL instance.

```bash
cd icarus-cdk/infra
npx cdk deploy --profile icarus
```

After deployment completes, CDK prints outputs like this:

```
Outputs:
IcarusDannerInfraStack.ElectionDbEndpoint = xxxxxxxx.xxxxxxxx.us-east-1.rds.amazonaws.com
IcarusDannerInfraStack.ElectionDbSecretArn = arn:aws:secretsmanager:us-east-1:123456789:secret:ElectionDataDbSecret-XXXXXX
```

Save both values — you'll need them for every step below.

## 2. Retrieve RDS Credentials

Use the `ElectionDbSecretArn` value from the CDK output:

```bash
aws secretsmanager get-secret-value \
  --secret-id "arn:aws:secretsmanager:us-east-1:123456789:secret:ElectionDataDbSecret-XXXXXX" \
  --profile icarus \
  --query SecretString --output text
```

This returns a JSON string like:

```json
{"username":"postgres","password":"GENERATED_PASSWORD","host":"xxxxxxxx.us-east-1.rds.amazonaws.com","port":5432,"dbname":"virginia_elections"}
```

Copy the `password` value — you'll use it in steps 3 and 5.

## 3. Run the Schema Script

Apply the DDL to create tables, indexes, and the `pg_trgm` extension. Use the `ElectionDbEndpoint` value from the CDK output as the host:

```bash
psql -h xxxxxxxx.xxxxxxxx.us-east-1.rds.amazonaws.com \
  -U postgres -d virginia_elections -f schema.sql
```

When prompted for a password, paste the `password` value from step 2.

## 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the ETL

In the commands below, replace the host and password with your actual values from steps 1 and 2.

### Local mode (default)

Reads JSON files from the `election-data/` directory:

```bash
python etl.py --source local \
  --db-url "postgresql://postgres:GENERATED_PASSWORD@xxxxxxxx.xxxxxxxx.us-east-1.rds.amazonaws.com:5432/virginia_elections"
```

### S3 mode (optional)

Reads JSON files from the `predictif-election-data` S3 bucket using the `icarus` profile:

```bash
python etl.py --source s3 \
  --db-url "postgresql://postgres:GENERATED_PASSWORD@xxxxxxxx.xxxxxxxx.us-east-1.rds.amazonaws.com:5432/virginia_elections"
```

## Idempotency

The ETL is safe to re-run. It checks each file's `record_id` against the `elections` table before inserting — duplicates are skipped automatically. Each file is processed in its own transaction, so a failure in one file does not affect others.
