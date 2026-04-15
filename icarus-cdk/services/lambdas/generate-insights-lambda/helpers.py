import json
import boto3
import os
from typing import (
    Dict,
    Optional,
    List,
    Tuple,
    Any,
    Set
)
from botocore.config import Config
from datetime import datetime, date
import tiktoken
from threading import Lock

# Initialize Boto3 Clients
config = Config(
    read_timeout=300,
    connect_timeout=60,
    retries={
        'total_max_attempts': 5,
        'mode': 'adaptive'
    }
)
bedrock_runtime = boto3.client("bedrock-runtime", config=config)
s3_client = boto3.client("s3")
sts_client = boto3.client("sts")
dynamodb = boto3.resource('dynamodb')

# Environment variables
INPUT_TOKEN_LIMIT = 200000
ACCOUNT_ID = sts_client.get_caller_identity()['Account']
INSIGHTS_GENERALISED_PROMPT = os.getenv("INSIGHTS_GENERALISED_PROMPT", 'campaign_insights_prompt.md')
KB_INSIGHTS_PROMPT = os.getenv("KB_INSIGHTS_PROMPT", "kb_election_laws_prompt.md")
PROMPT_BUCKET = os.getenv("PROMPT_BUCKET", 'prompt-bucket')
PROMPT_BUCKET = f"{PROMPT_BUCKET}-{ACCOUNT_ID}"
# TODO: Replace this with something more robust and worthy of MVP
ELECTION_CYCLE_FILENAME = os.getenv("ELECTION_CYCLE_FILENAME", 'election_cycles.json')
MODEL_ID = os.environ.get('MODEL_ID', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')
KB_ID = os.environ.get('KB_ID', '')
MAIN_TABLE_NAME = os.getenv("MAIN_TABLE_NAME")

main_table = dynamodb.Table(MAIN_TABLE_NAME) #type: ignore

response = s3_client.get_object(Bucket=PROMPT_BUCKET, Key=ELECTION_CYCLE_FILENAME)
ELECTION_CYCLES_DATA = json.loads(response["Body"].read().decode('utf-8'))

# Fields to ignore when extracting questionnaire answers from DynamoDB record
IGNORED_FIELDS = {'userId', 'savedAt', 'updatedAt'}

def error_response(status_code, message):
    """Helper function to return error responses."""
    return {
        'statusCode': status_code,
        'body': json.dumps({'error': message}),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }

def split_counter(query: str) -> Dict[str, int]:
    """
    This function determines how many concurrent LLM calls are needed 
    by counting how many splits it takes to pass through the model.

    Args:
        enriched_questions (List): List of dict containing section, question and context
        document_bytes (_type_): The application form template
        application_writing_prompt (str): The system prompt

    Returns:
        int: Number of split
    """
    # Start off with two
    split = 2
    # Flag to detemine when all splits have passed
    split_failed_flag = True
    remainder = 1
    split_tokens = []
    print(f"Total length of query: {len(query)}")
    while split_failed_flag:
        if split >= 4:
            break
        print(f"Split: {split}")
        parts, remainder = divmod(len(query), split)
        print(f"Parts: {parts}, Remainder: {remainder}")
        start_index = 0
        # Flag to determine if all splits passed or not
        counting_failed_flag = False
        for i in range(split):
            end_index = start_index + parts + (1 if i<remainder else 0)
            print(f"Start index: {start_index}, End index: {end_index}")
            query_subset = query[start_index: end_index]
            start_index = end_index

            # Count tokens and check if less than limit
            token_count = count_tokens(prompt=query_subset)
            print(f"Tokens of {i}th part: {token_count}")
            if not token_count:
                counting_failed_flag = True
                break
            else:
                split_tokens.append(token_count)
        # If split counting failed, increment split
        if counting_failed_flag == True:
            split += 1
        else:
            print(f"{split} splits work. Each split has {parts} with {remainder} remainder")
            return {
                "split": split,
                "parts": parts,
                "remainder": remainder
            }
    # Split counter failed
    return {
        "split": 0,
        "parts": 0,
        "remainder": 0
    }


def count_tokens(prompt: str) -> Optional[int]:
    # Encoding used by Anthropic
    try:
        response = bedrock_runtime.count_tokens(
            modelId=MODEL_ID,
            input={"converse": prompt}
        )
        return response["inputTokens"]
    except Exception as e:
        print("Counting tokens failed, N+1 strategy NEEDS to be implemented")
        return None

class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.lock = Lock()
    
    def increment_completed(self):
        with self.lock:
            self.completed += 1
            print(f"Progress: {self.completed}/{self.total} completed, {self.failed} failed")
    
    def increment_failed(self):
        with self.lock:
            self.failed += 1

class PCMChatbot():
    def __init__(self,
                 scope_model_id: str,
                 model_id: str,
                 s3_client: Any,
                 bedrock_runtime: Any,
                 bedrock_agent_runtime: Any,
                 candidate_context: Dict[str, str]):
        self.scope_model_id = scope_model_id
        self.model_id = model_id
        self.s3_client = s3_client
        self.bedrock_runtime = bedrock_runtime
        self.bedrock_agent_runtime = bedrock_agent_runtime
        self.candidate_context = candidate_context

    def load_data(self):
        print("Creating S3 paths of candidates election data and other office's election data")
        retrieval_plan = self.generate_retrieval_plan_from_election_cycles()
        try:
            candidate_election_data, other_election_data = self.extract_data_from_s3_all_elections(retrieval_plan=retrieval_plan,
                                                                    district_name=self.candidate_context['district_name'])
            return candidate_election_data, other_election_data
        except Exception as e:
            print(f"Error when extracting data from S3 paths: {str(e)}")
            return None, None

    def generate_retrieval_plan_from_election_cycles(
        self
    ) -> Dict[str, Any]:
        """
        Generate the retrieval plan automatically based on candidate context and election cycles.
        """
        candidate_office = self.candidate_context['office_position']
        district_name = self.candidate_context['district_name']
        current_year = date.today().year
        lookback_years = 5
        
        years_all = list(range(current_year - lookback_years, current_year + 1))
        presidential_years = [y for y in years_all if y % 4 == 0]
        
        retrieval_plan = {
            'candidate_office_data': [],
            'all_other_elections': [],
            'analysis_context': {
                'total_years_analyzed': lookback_years,
                'years_covered': years_all,
                'presidential_years_included': presidential_years,
                'total_elections_retrieved': 0,
                'offices_included': []
            },
            'reasoning': ''
        }
        
        # 1. Get candidate's office data
        candidate_years = self._get_election_years_in_window(candidate_office, current_year, lookback_years)
        
        for year in candidate_years:
            for election_type in ['Democratic_Primary', 'Republican_Primary', 'General_Election']:
                s3_path = f"{candidate_office}/{year}/{election_type}/{candidate_office}_{year}_{election_type}.json"
                
                retrieval_plan['candidate_office_data'].append({
                    's3_path': s3_path,
                    'extraction_spec': {
                        'level': 'DISTRICT_PRECINCT_DETAIL',
                        'filters': {
                            'district_name': district_name
                        }
                    },
                    'election_info': {
                        'year': year,
                        'election_type': election_type,
                        'office': candidate_office,
                        'is_presidential_year': (year in presidential_years),
                        'is_statewide': False
                    },
                    'purpose': f"{year} {candidate_office} {election_type} for {district_name}"
                })
        
        retrieval_plan['analysis_context']['offices_included'].append(candidate_office)
        
        # 2. Get election data from other offices
        for election_def in ELECTION_CYCLES_DATA['elections']:
            office_name = election_def['election']
            
            if office_name == candidate_office:
                continue
            
            office_election_years = self._get_election_years_in_window(office_name, current_year, lookback_years)
            
            for year in office_election_years:
                for election_type in ['Democratic_Primary', 'Republican_Primary', 'General_Election']:
                    s3_path = f"{office_name}/{year}/{election_type}/{office_name}_{year}_{election_type}.json"
                    
                    retrieval_plan['all_other_elections'].append({
                        's3_path': s3_path,
                        'extraction_spec': {
                            'level': 'DISTRICT_PRECINCT_DETAIL',
                            'filters': {
                                'district_name': 'Statewide' if election_def['is_statewide'] else district_name
                            }
                        },
                        'election_info': {
                            'year': year,
                            'election_type': election_type,
                            'office': office_name,
                            'is_presidential_year': (year in presidential_years),
                            'is_statewide': election_def['is_statewide']
                        },
                        'purpose': f"{year} {office_name} {election_type}" + 
                                    (f" (filtered to {district_name} precincts)" if election_def['is_statewide'] else f" for {district_name}")
                    })
            
            if office_election_years and office_name not in retrieval_plan['analysis_context']['offices_included']:
                retrieval_plan['analysis_context']['offices_included'].append(office_name)
        
        total_elections = len(retrieval_plan['candidate_office_data']) + len(retrieval_plan['all_other_elections'])
        retrieval_plan['analysis_context']['total_elections_retrieved'] = total_elections
        
        return retrieval_plan

    def extract_data_from_s3_all_elections(self, retrieval_plan: Dict[str, Any], district_name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract precinct-level data from S3 for all elections.
        For statewide races, filters to only precincts that exist in the candidate's district.
        """
        bucket_name = "predictif-election-data"
        
        candidate_office_data = []
        for plan_item in retrieval_plan.get('candidate_office_data', []):
            s3_path = plan_item['s3_path']
            extraction_spec = plan_item['extraction_spec']
            election_info = plan_item.get('election_info', {})
            purpose = plan_item.get('purpose', '')
            
            try:
                response = self.s3_client.get_object(Bucket=bucket_name, Key=s3_path)
                full_data = json.loads(response['Body'].read().decode('utf-8'))
                extracted_chunk = self._extract_district_precinct_detail(full_data, district_name)
                extracted_chunk['_metadata'] = {
                    'source_path': s3_path,
                    'category': 'candidate_office',
                    'purpose': purpose,
                    'election_info': election_info,
                    'extraction_level': 'DISTRICT_PRECINCT_DETAIL'
                }
                candidate_office_data.append(extracted_chunk)
            except Exception as e:
                print(f"Error extracting candidate office data from {s3_path}: {str(e)}")
        
        candidate_precincts = self._get_candidate_district_precincts(candidate_office_data)
        other_election_data = []
        
        for plan_item in retrieval_plan.get('all_other_elections', []):
            s3_path = plan_item['s3_path']
            extraction_spec = plan_item['extraction_spec']
            election_info = plan_item.get('election_info', {})
            purpose = plan_item.get('purpose', '')
            is_statewide = election_info.get('is_statewide', False)
            
            try:
                response = self.s3_client.get_object(Bucket=bucket_name, Key=s3_path)
                full_data = json.loads(response['Body'].read().decode('utf-8'))
                filter_district = extraction_spec.get('filters', {}).get('district_name', district_name)
                
                if is_statewide:
                    extracted_chunk = self._extract_statewide_filtered_to_district(
                        full_data, filter_district, candidate_precincts
                    )
                else:
                    extracted_chunk = self._extract_district_precinct_detail(full_data, filter_district)
                
                extracted_chunk['_metadata'] = {
                    'source_path': s3_path,
                    'category': 'other_elections',
                    'purpose': purpose,
                    'election_info': election_info,
                    'extraction_level': 'DISTRICT_PRECINCT_DETAIL',
                    'is_statewide': is_statewide
                }
                other_election_data.append(extracted_chunk)
                print(f"Extracted data from: {s3_path}")
            except Exception as e:
                print(f"Error extracting other election data from {s3_path}: {str(e)}")
        
        return candidate_office_data, other_election_data

    def get_scope_decision_from_bedrock(self,
                                        user_query: str,
                                        candidate_context: str) -> str:
        """
        Call Bedrock to determine if we need ALL_ELECTIONS or CANDIDATE_OFFICE_ONLY or None.
        This is a lightweight call that returns minimal JSON.

        Parameters:
            user_query (str): The user's query
            candidate_context (dict): The office position, district name/number and year they are running for.
            model_id (str): The model ID.

        Returns:
            scope (str): JSON string response in the following format\n
                        {\n
                            "scope": "CANDIDATE_OFFICE_ONLY" or "ALL_ELECTIONS",\n
                            "reasoning": "1-2 sentence explanation of why this scope was chosen"\n
                        }   
        """
        response = self.s3_client.get_object(Bucket=PROMPT_BUCKET, Key='scope_decision_prompt.txt')
        system_prompt = response["Body"].read().decode('utf-8')
        
        election_cycles_context = self._generate_election_cycles_context(5)
        
        full_prompt = system_prompt.replace('{{CANDIDATE_CONTEXT}}', candidate_context)
        full_prompt = full_prompt.replace('{{ELECTION_CYCLES}}', election_cycles_context)
        full_prompt = full_prompt.replace('{{USER_QUERY}}', user_query)
        
        response = self.bedrock_runtime.converse(
            modelId=self.model_id,
            messages=[
                {
                    'role': 'user',
                    'content': [{'text': full_prompt}]
                }
            ],
            inferenceConfig={
                'temperature': 0.1,
                'maxTokens': 500
            }
        )
        
        response_text = response['output']['message']['content'][0]['text']
        
        try:
            decision = json.loads(response_text)
            scope = decision.get('scope', 'ALL_ELECTIONS')
            print(f"Scope decision: {scope}")
            print(f"Reasoning: {decision.get('reasoning', 'N/A')}")
            return scope
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(1))
                return decision.get('scope', 'ALL_ELECTIONS')
            else:
                print("Could not parse scope decision, defaulting to ALL_ELECTIONS")
                return 'ALL_ELECTIONS'
            
    def create_llm_prompt_with_context_all_elections(self,
                                                     election_data: Optional[List[Dict[str, Any]]], 
                                                     candidate_context: str) -> str:
        """
        Create the final prompt for the LLM with all elections precinct-level context.
        """
        if election_data:
            print("Using Insights Generalized Prompt")
            context = self._format_for_llm_context_all_elections(election_data)
            print(f"Formatted DATA context: {context[:20]}")
            election_cycles_context = self._generate_election_cycles_context(5)

            try:
                print(f"S3 PATH: prompt-bucket-{ACCOUNT_ID}/{INSIGHTS_GENERALISED_PROMPT}")
                response = s3_client.get_object(Bucket=f'prompt-bucket-{ACCOUNT_ID}', Key=INSIGHTS_GENERALISED_PROMPT)
                insights_prompt = response['Body'].read().decode('utf-8')

                print(f"S3 PATH: prompt-bucket-{ACCOUNT_ID}/{KB_INSIGHTS_PROMPT}")
                response = s3_client.get_object(Bucket=f'prompt-bucket-{ACCOUNT_ID}', Key=KB_INSIGHTS_PROMPT)
                kb_insights_prompt = response['Body'].read().decode('utf-8')

                print("Getting election laws")
                election_laws = self._retrieve_laws(user_query=kb_insights_prompt)
                print("Got election laws")

                print(f"S3 prompt: {insights_prompt[:20]}")
                insights_prompt = insights_prompt.replace("{candidate_context}", candidate_context)
                insights_prompt = insights_prompt.replace("{election_cycles_context}", election_cycles_context)
                if election_laws:
                    insights_prompt = insights_prompt.replace("{election_laws}", election_laws)
                insights_prompt = insights_prompt.replace("{context}", context)
                print(f"Formatted prompt: {insights_prompt[:20]}")
                return insights_prompt
            except Exception as e:
                error_response(status_code=400, message=f"Not able to retrieve insights prompt: {str(e)}")
                return ''
        else:
            return ""
    
    def get_answer_from_bedrock(self, prompt: str, progress: Optional[ProgressTracker]) -> Dict:
        """
        Call Bedrock to get the final strategic answer.
        """
        message = {
                    'role': 'user',
                    'content': [{'text': prompt}]
                }
        messages = [message]
        try:
            response = self.bedrock_runtime.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig={
                    'temperature': 0.3,
                    'maxTokens': 8000
                }
            )
            if progress:
                progress.increment_completed()
            return response
        except Exception as e:
            print(f"N + 1 approach failed when generating part answers: {e}")
            if progress:
                progress.increment_failed()
            return {}
    
    def _retrieve_laws(self, user_query: str) -> str:
        response = self.bedrock_agent_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={
                'text': user_query
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': 15
                }
            }
        )
        if response['retrievalResults']:
            results = [result['content']['text'] for result in response['retrievalResults']]
            return "\n\n".join(results)
        else:
            return ''

    def _format_for_llm_context_all_elections(self, extracted_data: List[Dict[str, Any]]) -> str:
        """
        Format extracted election data for LLM consumption - all precinct level.
        """

        def _format_precinct_chunk(data: Dict[str, Any]) -> str:
            """
            Format precinct-level data for a single election.
            """
            metadata = data.get('_metadata', {})
            election_info = metadata.get('election_info', {})
            is_statewide = metadata.get('is_statewide', False)
            parts = []
            
            year = data.get('year', 'N/A')
            stage = data.get('stage', 'N/A')
            office = election_info.get('office', data.get('office', 'Unknown'))
            is_pres_year = " (PRESIDENTIAL YEAR)" if election_info.get('is_presidential_year') else ""
            statewide_note = " [Filtered to District Precincts]" if is_statewide else ""
            
            parts.append("\n" + "─" * 80)
            parts.append(f"📊 {office} {year} {stage}{is_pres_year}{statewide_note}")
            parts.append("─" * 80)
            parts.append(f"Purpose: {metadata.get('purpose', 'N/A')}")
            
            district = data.get('district')
            if not district:
                parts.append("⚠️  No district data available")
                return "\n".join(parts)
            
            parts.append("")
            parts.append(f"District: {district.get('district_name', 'Unknown')}")
            parts.append(f"  📍 District Total Votes: {district.get('district_total_votes', 0):,}")
            
            if is_statewide:
                parts.append(f"  ℹ️  Note: Statewide race filtered to show only precincts in your district")
            
            if district.get('district_win_number') is not None:
                parts.append(f"  🎯 Votes Needed to Win: {district.get('district_win_number'):,}")
            
            if district.get('district_flip_number') is not None:
                parts.append(f"  🔄 Votes Needed to Flip: {district.get('district_flip_number'):,}")

            if district.get('district_win_gap') is not None:
                parts.append(f"  Win Gap between winner and runner up: {district.get('district_win_gap'):,}")

            district_total = district.get('district_total_votes', 0)
            district_results = district.get('district_results')
            if district_results:
                results_sorted = sorted(district_results, key=lambda x: x.get('votes', 0), reverse=True)
                for i, candidate in enumerate(district_results, 1):
                    candidate_name = candidate.get('candidate_name', 'Unknown')
                    votes = candidate.get('votes', 0)
                    percentage = (votes / district_total * 100) if district_total > 0 else 0
                    indicator = "🏆" if i == 1 else "  "
                    parts.append(f"       {indicator} {i}. {candidate_name}: {votes:,} votes ({percentage:.1f}%)")
                
                if len(results_sorted) >= 2:
                    margin = results_sorted[0].get('votes', 0) - results_sorted[1].get('votes', 0)
                    margin_pct = (margin / district_total * 100) if district_total > 0 else 0
                    parts.append(f"     Margin: {margin:,} votes ({margin_pct:.1f}%)")
            
            parts.append("")
            parts.append("PRECINCT BREAKDOWN:")
            parts.append("")
            
            precincts = district.get('precincts', [])
            precincts_sorted = sorted(precincts, key=lambda x: x.get('precinct_total_votes', 0), reverse=True)
            
            for i, precinct in enumerate(precincts_sorted, 1):
                precinct_name = precinct.get('precinct_name', 'Unknown')
                precinct_total = precinct.get('precinct_total_votes', 0)
                
                parts.append(f"  {i}. {precinct_name}")
                parts.append(f"     Total Votes: {precinct_total:,}")
                
                if precinct.get('win_number') is not None:
                    parts.append(f"     Win Number: {precinct.get('win_number'):,}")
                if precinct.get('flip_number') is not None:
                    parts.append(f"     Flip Number: {precinct.get('flip_number'):,}")
                if precinct.get('win_gap') is not None:
                    parts.append(f"     Win Gap between winner and runner up: {precinct.get('win_gap'):,}")
                
                results = precinct.get('results', [])
                if results:
                    results_sorted = sorted(results, key=lambda x: x.get('votes', 0), reverse=True)
                    parts.append(f"     Candidate Results:")
                    
                    for j, result in enumerate(results_sorted, 1):
                        candidate = result.get('candidate_name', 'Unknown')
                        votes = result.get('votes', 0)
                        percentage = (votes / precinct_total * 100) if precinct_total > 0 else 0
                        indicator = "🏆" if j == 1 else "  "
                        parts.append(f"       {indicator} {j}. {candidate}: {votes:,} votes ({percentage:.1f}%)")
                    
                    if len(results_sorted) >= 2:
                        margin = results_sorted[0].get('votes', 0) - results_sorted[1].get('votes', 0)
                        margin_pct = (margin / precinct_total * 100) if precinct_total > 0 else 0
                        parts.append(f"     Margin: {margin:,} votes ({margin_pct:.1f}%)")
                
                parts.append("")
            
            total_district_votes = district.get('district_total_votes', 0)
            if total_district_votes > 0 and precincts:
                parts.append("DISTRICT SUMMARY:")
                parts.append(f"  • Total precincts: {len(precincts)}")
                parts.append(f"  • Total district votes: {total_district_votes:,}")
                parts.append(f"  • Average votes per precinct: {total_district_votes // len(precincts):,}")
            
            parts.append("")
            return "\n".join(parts)
        
        context_parts = []
        context_parts.append("=" * 80)
        context_parts.append("COMPREHENSIVE ELECTION DATA ANALYSIS FOR YOUR CAMPAIGN")
        context_parts.append("=" * 80)
        context_parts.append("")
        context_parts.append("DATA SCOPE:")
        context_parts.append("  • 5 years of historical precinct-level data")
        context_parts.append("  • All elections that occurred in your district during this period")
        context_parts.append("  • Includes primaries and general elections for all offices")
        context_parts.append("  • For statewide races: ONLY precincts in your district")
        context_parts.append("")
        context_parts.append("=" * 80)
        context_parts.append("")
        
        candidate_data = [d for d in extracted_data if d.get('_metadata', {}).get('category') == 'candidate_office']
        other_data = [d for d in extracted_data if d.get('_metadata', {}).get('category') == 'other_elections']
        
        candidate_data.sort(key=lambda x: x.get('year', 0), reverse=True)
        other_data.sort(key=lambda x: x.get('year', 0), reverse=True)
        
        if candidate_data:
            context_parts.append(f"YOUR OFFICE ELECTION HISTORY:")
            context_parts.append("=" * 80)
            context_parts.append("Precinct-level data for your office over the past 5 years")
            context_parts.append("")
            for data in candidate_data:
                context_parts.append(_format_precinct_chunk(data))
            context_parts.append("")
        
        if other_data:
            context_parts.append("ALL OTHER ELECTIONS IN YOUR DISTRICT:")
            context_parts.append("=" * 80)
            context_parts.append("Precinct-level data for all other offices - SAME PRECINCTS as your district")
            context_parts.append("")
            for data in other_data:
                context_parts.append(_format_precinct_chunk(data))
            context_parts.append("")
        
        return "\n".join(context_parts)

    def _generate_election_cycles_context(self, lookback_years: int) -> str:
        """
        Generate a human-readable description of which elections occur in the past N years.
        """
        current_year = datetime.now().year
        years = list(range(current_year - lookback_years, current_year + 1))
        
        context_lines = []
        context_lines.append(f"Looking back {lookback_years} years from {current_year}:")
        context_lines.append("")
        
        for election in ELECTION_CYCLES_DATA['elections']:
            office_name = election['election']
            cycle = election['cycle']
            pattern = election['election_pattern']
            
            elections_in_window = []
            
            if pattern == 'even':
                elections_in_window = [y for y in years if y % 2 == 0]
            elif pattern == 'odd':
                elections_in_window = [y for y in years if y % 2 == 1]
            elif pattern == 'even_biennial':
                elections_in_window = [y for y in years if y % 2 == 0]
            elif pattern == 'annual':
                elections_in_window = years
            elif pattern == 'periodic':
                elections_in_window = [y for y in years if (y - current_year) % cycle == 0]
            
            if elections_in_window:
                context_lines.append(f"• {office_name} ({pattern}): {', '.join(map(str, elections_in_window))}")
        
        return "\n".join(context_lines)
    
    def _get_election_years_in_window(self, office_name: str, current_year: int, lookback_years: int = 5) -> List[int]:
        """
        Determine which years in the lookback window have elections for a given office.
        
        Args:
            office_name: Name of the office (e.g., "House_of_Delegates", "President")
            current_year: Current year
            lookback_years: How many years to look back
        
        Returns:
            List of years when elections occurred
        """
        years = list(range(current_year - lookback_years, current_year + 1))
        
        election_def = None
        for election in ELECTION_CYCLES_DATA['elections']:
            if election['election'] == office_name:
                election_def = election
                break
        
        if not election_def:
            print(f"Office {office_name} not found in election cycles")
            return []
        
        cycle = election_def['cycle']
        pattern = election_def['election_pattern']
        
        elections_in_window = []
        
        if pattern == 'even':
            elections_in_window = [y for y in years if y % 2 == 0]
        elif pattern == 'odd':
            elections_in_window = [y for y in years if y % 2 == 1]
        elif pattern == 'even_biennial':
            elections_in_window = [y for y in years if y % 2 == 0]
        elif pattern == 'annual':
            elections_in_window = years
        elif pattern == 'periodic':
            elections_in_window = []
            for y in years:
                if y % cycle == current_year % cycle:
                    elections_in_window.append(y)
        
        print(f"Election years for {office_name}: {elections_in_window}")
        return elections_in_window

    def _get_candidate_district_precincts(self, extracted_candidate_data: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract the set of precinct names from the candidate's district data.
        This will be used to filter statewide races to only relevant precincts.
        """
        precinct_names = set()
        
        for data in extracted_candidate_data:
            district = data.get('district')
            if district:
                precincts = district.get('precincts', [])
                for precinct in precincts:
                    precinct_name = precinct.get('precinct_name')
                    if precinct_name:
                        precinct_names.add(precinct_name)
        
        print(f"Found {len(precinct_names)} unique precincts in candidate's district")
        return precinct_names
    
    def _extract_district_precinct_detail(self, data: Dict, district_name: str) -> Dict:
        """
        Extract full precinct-level detail for a specific district.
        Works for district-based races (District_3).
        """
        result = {
            'record_id': data.get('record_id'),
            'office': data.get('office'),
            'year': data.get('year'),
            'stage': data.get('stage'),
            'total_votes': data.get('total_votes'),
            'district': None
        }
        
        for district in data.get('districts', []):
            if district.get('district_name') == district_name:
                result['district'] = district
                break
        
        if result['district'] is None:
            result['_error'] = f"District {district_name} not found in data"
        
        return result

    def _extract_statewide_filtered_to_district(self, data: Dict, district_name: str, candidate_precincts: Set[str]) -> Dict:
        """
        Extract statewide race data (Governor, President) but filter to only precincts
        that exist in the candidate's district.
        
        Args:
            data: Full statewide election data
            district_name: Should be "Statewide"
            candidate_precincts: Set of precinct names from candidate's district
        """
        result = {
            'record_id': data.get('record_id'),
            'office': data.get('office'),
            'year': data.get('year'),
            'stage': data.get('stage'),
            'total_votes': data.get('total_votes'),
            'district': None
        }
        
        statewide_district = None
        for district in data.get('districts', []):
            if district.get('district_name') == district_name:
                statewide_district = district
                break
        
        if statewide_district is None:
            result['_error'] = f"District {district_name} not found in data"
            return result
        
        filtered_precincts = []
        all_precincts = statewide_district.get('precincts', [])
        
        for precinct in all_precincts:
            precinct_name = precinct.get('precinct_name')
            if precinct_name in candidate_precincts:
                filtered_precincts.append(precinct)
        
        print(f"Filtered statewide {data.get('office')} {data.get('year')} {data.get('stage')}: "
                    f"{len(all_precincts)} total precincts -> {len(filtered_precincts)} matching candidate's district")
        
        filtered_district = {
            'district_name': district_name,
            'district_total_votes': sum(p.get('precinct_total_votes', 0) for p in filtered_precincts),
            'district_win_number': None,
            'district_flip_number': None,
            'precincts': filtered_precincts
        }
        
        result['district'] = filtered_district
        
        if not filtered_precincts:
            result['_warning'] = f"No matching precincts found in statewide data for candidate's district"
        
        return result
