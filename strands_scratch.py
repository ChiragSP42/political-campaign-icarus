from strands import Agent, tool
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from typing import Optional, List, Dict
from dotenv import load_dotenv
load_dotenv(override=True)

def get_db_connection():
    """
    Establish connection to PostgreSQL server hosted in RDS
    """

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )

@tool
def election_record(year: Optional[str]=None, office: Optional[str]=None, stage: Optional[str]=None):
    """
    Get a specific election record. You can filter by year, type of election and stage.
    For example, if you want to query all the elections for the year 2021 the arguments

    Args:
        year: year of election
        office: election office (Governor, House_of_Delegates, Town_Council, etc.)
        stage: whether General_Election, Democratic_Primary or Republican_Primary

    Returns:
        List[Dict] result of query 
    """

    query = """SELECT id, record_id, year, office, stage, total_votes FROM elections WHERE"""
    kwargs = {"year": year, "office": office, "stage": stage}
    conn = get_db_connection()
    kwargs_cleaned = {k:v for k,v in kwargs.items() if v}
    print(kwargs_cleaned)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for idx, (key, value) in enumerate(kwargs_cleaned.items()):
                if value:
                    query = f"{query} {key} = %s"
                    if idx != len(kwargs_cleaned.keys())-1:
                        query = f"{query} AND"
            query = f"{query};"
            print(query)

            cur.execute(query, tuple([kwarg for kwarg in kwargs_cleaned.values()]))

            return cur.fetchall()
    except Exception as e:
        print(f"Error: {e}")


@tool
def district_record(year: Optional[str]=None, office: Optional[str]=None, district_name: Optional[str]=None):
    """
    Fetches the competitive metrics (win number, flip number, win gap) and total turnout for a specific district within a given election context.

    Args:
        year: year of election
        office: election office (Governor, House_of_Delegates, Town_Council, etc.)
        district_name: name of the district/county

    Returns:
        List[Dict] result of query 
    """

    query = """SELECT d.district_name, d.total_votes, d.win_number, d.flip_number, d.win_gap 
FROM districts d 
JOIN elections e ON d.election_id = e.id 
WHERE"""
  
    kwargs = {"e.year": year, "e.office": office, "d.district_name": district_name}
    conn = get_db_connection()
    kwargs_cleaned = {k:v for k,v in kwargs.items() if v}
    print(kwargs_cleaned)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for idx, (key, value) in enumerate(kwargs_cleaned.items()):
                if value:
                    query = f"{query} {key} = %s"
                    if idx != len(kwargs_cleaned.keys())-1:
                        query = f"{query} AND"
            query = f"{query};"
            print(query)

            cur.execute(query, tuple([kwarg for kwarg in kwargs_cleaned.values()]))

            return cur.fetchall()
    except Exception as e:
        print(f"Error: {e}")

agent = Agent(tools=[election_record, district_record])

# Testing election_record tool
# response = agent("What was the total number of votes for House of Delegates in the year 2021?")

# Testing district_record tool
response = agent("What was the total number of votes in District_1")

# print(response)