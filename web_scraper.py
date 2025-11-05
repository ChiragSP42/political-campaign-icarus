#%%
from typing import List, Dict, Literal, Optional
import os
import json
from tqdm.auto import tqdm
import boto3
import time
import pandas as pd
import numpy as np
import logging
import requests
from datetime import date
from aws_helpers import helpers
import re
from io import StringIO
from dotenv import load_dotenv
from tavily import TavilyClient
load_dotenv(override=True)

logger = helpers._setup_logger(name="election-scrapper", level=logging.DEBUG)

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

# Extract type of election and district name from filename------
def content_extraction(filename, is_statewide, office_position):
    if not is_statewide:
        district_number = 0
        election_type = ''
        match = re.search(r'_([A-Za-z]+(?:_[A-Za-z]+)?)_District_(\d+)', filename)
        if match:
            election_type = match.group(1)
            district_number = match.group(2)

            return district_number, election_type
        else:
            return None, None
    else:
        match = re.search(r'_([A-Za-z]+(?:_[A-Za-z]+)?)_including_precincts', filename)
        if match:
            election_type = match.group(1)

            return None, election_type
        else:
            return None, None

# Extract data and populate JSON----------------
def data_population(year: int, df, office_position: str, is_statewide: bool) -> Dict:        
    cols = df.columns
    winner_name = cols[3]
    filename = df.attrs['source_file']
    district_number, election_type = content_extraction(filename=filename, office_position=office_position, is_statewide=is_statewide)
    if district_number is None:
        district_number = 'Statewide'
    else:
        district_number = f'District_{district_number}'
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
    district = {}
    for index, row in df.iterrows():
        if row['County/City'] == 'TOTALS':
            continue
        precinct = {}
        if index == 0:
            continue
        else:
            # logger.debug(f"Row:\n{row}")
            # logger.debug(f"Pct: {row['Pct']}\nWinner: {row[winner_name]}")
            results = []
            for i in range(3, 3+(len(df.columns)-6) + 1):
                candidate = {
                    "candidate_name": cols[i],
                    "votes": row[cols[i]]
                }
                results.append(candidate)

            if len(df.columns) == 6:
                precinct = {
                    "precinct_name": f"{row["County/City"]}_{row['Pct']}",
                    "precinct_total_votes": row['Total Votes Cast'],
                    "results": results,
                    "win_number": np.ceil((row['Total Votes Cast'] / 2) + 1),
                    "flip_number": 0
                }
            else:
                runner_up = cols[4]
                precinct = {
                    "precinct_name": f"{row["County/City"]}_{row['Pct']}",
                    "precinct_total_votes": row['Total Votes Cast'],
                    "results": results,
                    "win_number": np.ceil((row['Total Votes Cast'] / 2) + (abs(row[winner_name] - row[runner_up]) / 2) + 1),
                    "flip_number": np.ceil((abs(row[winner_name] - row[runner_up]) / 2) + 1)
                }
        precincts.append(precinct)
    if len(df.columns) == 6:
        district = {
            "district_name": f"{district_number}",
            "district_total_votes": pd.to_numeric(df.loc[(df.shape[0] - 1), "Total Votes Cast"], downcast='integer'),
            "district_win_number": np.ceil((df.loc[(df.shape[0] - 1), "Total Votes Cast"] / 2) + 1),
            "district_flip_number": 0,
            "precincts": precincts
        }
    else:
        runner_up = cols[4]
        district = {
            "district_name": f"{district_number}",
            "district_total_votes": pd.to_numeric(df.loc[(df.shape[0] - 1), "Total Votes Cast"], downcast='integer'),
            "district_win_number": np.ceil((df.loc[(df.shape[0] - 1), "Total Votes Cast"] / 2) + (abs(df.loc[(df.shape[0] - 1), winner_name] - df.loc[(df.shape[0] - 1), runner_up]) / 2) + 1),
            "district_flip_number": np.ceil((abs(df.loc[(df.shape[0] - 1), winner_name] - df.loc[(df.shape[0] - 1), runner_up]) / 2) + 1),
            "precincts": precincts
        }

    return district

# S3 storage-------------         
def s3_storage(complete_data):
    path = f"{complete_data['office']}/{complete_data['year']}/{complete_data['stage']}/{complete_data['office']}_{complete_data['year']}_{complete_data['stage']}.json"
    s3_client.put_object(Bucket=S3_BUCKET,
                         Key=path,
                         Body=json.dumps(complete_data, indent=2),
                         ContentType='application/json')

# Get election years for office position------------------
def get_election_years_in_window(election: dict, current_year: int, lookback_years: int = 5) -> List[int]:
    """
    Determine which years in the lookback window have elections for a given office.
    
    Args:
        election (dict): Election dict containing name, cycle, election_pattern and description
        current_year: Current year
        lookback_years: How many years to look back
    
    Returns:
        List of years when elections occurred
    """
    years = list(range(current_year - lookback_years, current_year + 1))
    
    cycle = election['cycle']
    pattern = election['election_pattern']
    
    elections_in_window = []
    
    if pattern == 'even':
        # Even-numbered years
        elections_in_window = [y for y in years if y % 2 == 0]
    elif pattern == 'odd':
        # Odd-numbered years
        elections_in_window = [y for y in years if y % 2 == 1]
    elif pattern == 'even_biennial':
        # Senate: every 2 years in even years
        elections_in_window = [y for y in years if y % 2 == 0]
    elif pattern == 'annual':
        # Annual elections
        elections_in_window = years
    elif pattern == 'periodic':
        # Periodic elections (every N years)
        # Find the most recent election year
        elections_in_window = []
        for y in years:
            if y % cycle == current_year % cycle:
                elections_in_window.append(y)
    
    logger.info(f"Election years for {election['election']}: {elections_in_window}")
    return elections_in_window

# Process tavily response-----------------
def process_tavily_response(results, OFFICE_POSITION, YEAR, is_statewide):
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
        district_number, election_type = content_extraction(filename=filename, office_position=OFFICE_POSITION, is_statewide=is_statewide)
        if district_number == None and election_type == None:
            logger.info(f"\x1b[31mCould not extract district number/election type from filename {filename}\x1b[0m")
            continue

        district = data_population(year=YEAR,
                        df=df,
                        office_position=OFFICE_POSITION,
                        is_statewide=is_statewide)
        if election_type not in stages.keys():
            stages[election_type] = [district]
        else:
            stages[election_type].append(district)
    
    return stages
    
def main():
    # YEARS = [2020, 2021, 2022, 2023, 2024]
    # YEARS = [2020]
    # OFFICE_POSITION = 'House_of_Delegates'
    # OFFICE_POSITION = 'Lieutenant_Governor'
    # OFFICE_POSITION = 'U.S._Senate'
    # OFFICE_POSITION = 'U.S._House'
    # OFFICE_POSITION = 'Governor'
    current_year = date.today().year
    with open("election_cycle_testing.json", "r") as f:
        elections = json.loads(f.read())

    for election in elections['elections']: 
        OFFICE_POSITION = election['election']
        logger.info(f"\x1b[33mOffice position: {OFFICE_POSITION}\x1b[0m")
        OFFICE_ID = election['id']
        YEARS = get_election_years_in_window(election=election, current_year=current_year)
        is_statewide=election['is_statewide']
        for YEAR in YEARS:
            logger.info(f"\x1b[33mYear: {YEAR}\x1b[0m")
            # Check if election has already been populated to conserve tavily credits
            s3_elections = helpers.list_obj_s3(s3_client=s3_client,
                                               bucket_name=S3_BUCKET,
                                               folder_name='',
                                               delimiter='/')
            s3_elections = [e.replace('/', '') for e in s3_elections]
            # logger.debug(s3_elections)
            if OFFICE_POSITION in s3_elections:
                election_years = helpers.list_obj_s3(s3_client=s3_client,
                                                     bucket_name=S3_BUCKET,
                                                     folder_name=OFFICE_POSITION+"/",
                                                     delimiter='/')
                election_years = [int(y.replace('/', '').replace(OFFICE_POSITION, '')) for y in election_years]
                # logger.debug(election_years)
                if YEAR in election_years:
                    continue

            url=f"https://historical.elections.virginia.gov/elections/search/year_from:{YEAR}/year_to:{YEAR}/office_id:{OFFICE_ID}"
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
            
            stages = process_tavily_response(results=results, OFFICE_POSITION=OFFICE_POSITION, YEAR=YEAR, is_statewide=is_statewide)
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
            logger.info("*" * 80)
            time.sleep(60)
        logger.info("-" * 80)

if __name__ == "__main__":
    main()