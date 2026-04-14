from dotenv import load_dotenv
from tavily import TavilyClient

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)

client = TavilyClient(api_key=TAVILY_API_KEY)

YEAR = 2020
OFFICE_POSITION = 'House_of_Delegates'

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

# url=f"https://historical.elections.virginia.gov/elections/search/year_from:{YEAR}/year_to:{YEAR}/office_id:{OFFICE_ID}"
# url=f"https://historical.elections.virginia.gov/elections/search?df={YEAR}&dt={YEAR}&t=table&bq=false&coff[0].i={OFFICE_ID}"
url=f"https://historical.elections.virginia.gov/search?"
instructions=f"Get only the election data for {OFFICE_POSITION} from {YEAR} year at the precinct level as a downloadable csv"
logger.info("\x1b[33mBeginning crawl\x1b[0m")
results = crawl(url=url,
                instructions=instructions,
                limit=200,
                max_depth=3,
                max_breadth=200,
                extract_depth="advanced",
                allow_external=False
                )

