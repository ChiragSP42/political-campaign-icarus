from typing import List, Dict, Literal, Optional
import os
import json
from tqdm.auto import tqdm
import math
import boto3
import time
import pandas as pd
import logging
import requests
from aws_helpers import helpers
import re
from io import StringIO
from dotenv import load_dotenv
from tavily import TavilyClient
load_dotenv(override=True)

logger = helpers._setup_logger(name="idk", level=logging.DEBUG)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", None)
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", None)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)
S3_BUCKET = 'predictif-election-data'

# Define clients---------
logger.info("\x1b[33mCreating Tavily and BOTO3 clients\x1b[0m")
client = TavilyClient(api_key=TAVILY_API_KEY)
session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY,
                        region_name='us-east-1')
s3_client = session.client("s3")

# Crawl-------------
def crawl(url, 
          instructions, 
          limit: int=2, 
          max_depth: int=3, 
          max_breadth: int=2, 
          extract_depth: Literal['basic', 'advanced']='advanced', 
          allow_external: bool=False,
          max_retries: int=3) -> Optional[List[Dict]]:
    

    for attempt in range(max_retries):
        try:
            response = client.crawl(
                url=url,
                instructions=instructions,
                limit=limit,
                max_depth=max_depth,
                max_breadth=max_breadth,
                extract_depth=extract_depth,
                allow_external=allow_external
            )

            results = response['results']
            return results
        except TimeoutError as e:
            if attempt < max_retries - 1:
                wait_time = (5 ** attempt)
                logger.info(f"\x1b[31mTimeout on attempt {attempt + 1}. Retrying in {wait_time}s...\x1b[0m")
                time.sleep(wait_time)
            else:
                logger.info(f"\x1b[31mFailed after {max_retries} attempts\x1b[0m")
                raise

def content_extraction(filename, office_position):
    if office_position == "House_of_Delegates":
        district_number = 0
        election_type = ''
        match = re.search(r'_([A-Za-z]+(?:_[A-Za-z]+)?)_District_(\d+)', filename)
        if match:
            election_type = match.group(1)
            district_number = match.group(2)

            return district_number, election_type
        else:
            return None, None
    elif office_position == "Governor":
        return None, None
    else:
        return None, None


def district_data_population(year: int, df: pd.DataFrame, office_position: str):
    def precinct_no_contest_data_population(df):
        cols = df.columns
        winner_name = cols[3]
        filename = df.attrs['source_file']
        district_number, election_type = content_extraction(filename=filename, office_position=office_position)
        log =f"""
Filename: {filename}
Election type: {election_type}
District number: {district_number}
District_total_votes: {df.loc[(df.shape[0] - 1), "Total Votes Cast"]}
Year: {year}
Office Position: {office_position}
Winner: {winner_name}
    """
        # logger.debug(log)
        precincts = []
        for index, row in df.iterrows():
            precinct = {}
            if index == 0:
                continue
            else:
                # logger.debug(f"Row:\n{row}")
                # logger.debug(f"Pct: {row['Pct']}\nWinner: {row[winner_name]}")

                results = [
                    {
                        "candidate_name": winner_name,
                        "votes": row[winner_name]
                    }
                ]
                precinct = {
                    "precinct_name": row['Pct'],
                    "precinct_total_votes": row['Total Votes Cast'],
                    "results": results,
                    "win_number": math.ceil((row['Total Votes Cast'] / 2) + 1),
                    "flip_number": 0
                }
            precincts.append(precinct)
        
        district = {
            "district_name": f"District_{district_number}",
            "district_total_votes": df.loc[(df.shape[0] - 1), "Total Votes Cast"],
            "district_win_number": math.ceil((df.loc[(df.shape[0] - 1), "Total Votes Cast"] / 2) + 1),
            "district_flip_number": 0,
            "precincts": precincts
        }

        return district
    
    def precinct_data_population(df):
        cols = df.columns
        winner_name = cols[3]
        runner_up = cols[4]
        filename = df.attrs['source_file']
        district_number, election_type = content_extraction(filename=filename, office_position=office_position)
        log =f"""
Filename: {filename}
Election type: {election_type}
District number: {district_number}
District_total_votes: {df.loc[(df.shape[0] - 1), "Total Votes Cast"]}
Year: {year}
Office Position: {office_position}
Winner: {winner_name}
Runner_up: {runner_up}
    """
        # logger.debug(log)
        precincts = []
        for index, row in df.iterrows():
            precinct = {}
            if index == 0:
                continue
            else:
                # logger.debug(f"Row:\n{row}")
                # logger.debug(f"Pct: {row['Pct']}\nWinner: {row[winner_name]}")

                results = [
                    {
                        "candidate_name": winner_name,
                        "votes": row[winner_name]
                    },
                    {
                        "candidate_name": runner_up,
                        "votes": row[runner_up]
                    }
                ]
                precinct = {
                    "precinct_name": row['Pct'],
                    "precinct_total_votes": row['Total Votes Cast'],
                    "results": results,
                    "win_number": math.ceil((row['Total Votes Cast'] / 2) + (abs(row[winner_name] - row[runner_up]) / 2) + 1),
                    "flip_number": math.ceil((abs(row[winner_name] - row[runner_up]) / 2) + 1)
                }
            precincts.append(precinct)
        
        district = {
            "district_name": f"District_{district_number}",
            "district_total_votes": int(df.loc[(df.shape[0] - 1), "Total Votes Cast"]),
            "district_win_number": math.ceil((df.loc[(df.shape[0] - 1), "Total Votes Cast"] / 2) + (abs(df.loc[(df.shape[0] - 1), winner_name] - df.loc[(df.shape[0] - 1), runner_up]) / 2) + 1),
            "district_flip_number": math.ceil((abs(df.loc[(df.shape[0] - 1), winner_name] - df.loc[(df.shape[0] - 1), runner_up]) / 2) + 1),
            "precincts": precincts
        }

        return district

    # No contest for this district election
    if len(df.columns) == 6:
        district = precinct_no_contest_data_population(df=df)
        # logger.debug(json.dumps(precinct_no_contest(df=df), indent=2))
        return district
        # districts.append(no_contest(df=df))
    # At least two candidates in district election
    else:
        district = precinct_data_population(df=df)
        return district
         
def s3_storage(complete_data):
    path = f"{complete_data['office']}/{complete_data['year']}/{complete_data['stage']}/{complete_data['office']}_{complete_data['year']}_{complete_data['stage']}.json"
    s3_client.put_object(Bucket=S3_BUCKET,
                         Key=path,
                         Body=json.dumps(complete_data),
                         ContentType='application/json')
def main():
    YEARS = [2021]
    # OFFICE_POSITION = 'House_of_Delegates'
    OFFICE_POSITION = 'Governor'
    with open("mapping.json", "r") as f:
        OFFICE_MAP = json.loads(f.read())
    for YEAR in YEARS:
        logger.info(f"\x1b[33mGetting info {YEAR} year\x1b[0m")
        df = pd.DataFrame()
        url=f"https://historical.elections.virginia.gov/elections/search/year_from:{YEAR}/year_to:{YEAR}/office_id:{OFFICE_MAP["office_id"][OFFICE_POSITION]}"
        instructions="Get only the election data at the precinct level as a downloadable csv"
        logger.info("\x1b[33mBeginning crawl\x1b[0m")
        results = crawl(url=url,
                        instructions=instructions,
                        limit=200,
                        max_depth=3,
                        max_breadth=200,
                        extract_depth="advanced",
                        allow_external=False
                        )
        # Loop through districts for particular election year, office position and stage.
        stages = {}
        logger.info("\x1b[33mProcessing results\x1b[0m")
        for result in tqdm(results):
            url =result['url']
            logger.info(f"\x1b[33mProcessing URL: {url}\x1b[0m")
            if "https://historical.elections.virginia.gov/elections/download" not in url:
                continue

            # Make HTTP request
            response = requests.get(url)

            # Extract filename from Content-Disposition header
            filename = None
            if 'Content-Disposition' in response.headers:
                content_disposition = response.headers['Content-Disposition']
                # Parse filename from header (e.g., "attachment; filename=election_HOD_2023_General.csv")
                filename_match = re.findall('filename="?([^"]+)"?', content_disposition)
                filename = ''
                if filename_match:
                    filename = filename_match[0]

            # Load CSV content into DataFrame
            csv_content = StringIO(response.text)
            df = pd.read_csv(csv_content, header=0)
            # Reset index and change column type to float for vote related columns.
            df.reset_index(drop=True, inplace=True)
            for col in df.columns[3:]:
                df[col] = pd.to_numeric(
                                    df[col].astype(str).str.replace(',', ''),
                                    errors='coerce'  # Converts invalid values to NaN
                        )
            df.attrs['source_file'] = filename
            if not filename:
                logger.info(filename)
                logger.info("\x1b[31mFilename not found\x1b[0m")
                continue
            district_number, election_type = content_extraction(filename=filename, office_position=OFFICE_POSITION)
            if not district_number or not election_type:
                logger.info("\x1b[31mCould not extract district number/election type from filename\x1b[0m")
                continue

            district = district_data_population(year=YEAR,
                            df=df,
                            office_position=OFFICE_POSITION)
            if election_type not in stages.keys():
                stages[election_type] = [district]
            else:
                stages[election_type].append(district)

        for stage, districts in stages.items():
            complete_data = {
                "record_id": f"{OFFICE_POSITION}_{YEAR}_{stage}",
                "year": YEAR,
                "office": OFFICE_POSITION,
                "stage": stage,
                "total_votes": sum(d['district_total_votes'] for d in districts),
                "districts": districts
            }
            s3_storage(complete_data=complete_data)
            # logger.debug(json.dumps(complete_data, indent=2))
        time.sleep(60)

if __name__ == "__main__":
    main()
    # logger.debug(df.shape[0])
    # logger.debug(df.dtypes)
    # df.to_csv(df.attrs['source_file'], index=False)