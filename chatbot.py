from aws_helpers import helpers, utils
from typing import Dict, List, Any, Set, Tuple, Optional
from dotenv import load_dotenv
import os
import datetime
import json
import boto3
from botocore.config import Config
load_dotenv(override=True)

ELECTION_CYCLE_FILENAME = os.getenv("ELECTION_CYCLE_FILENAME", '')
INSIGHTS_GENERALISED_PROMPT = os.getenv("INSIGHTS_GENERALISED_PROMPT", '')
GENERAL_STRATEGY_PROMPT = os.getenv("GENERAL_STRATEGY_PROMPT", '')
PROMPT_FOLDER = 'texts'
KB_ID = os.getenv("KB_ID", '')
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", None)
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", None)

if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("AWS credentials not found")
if not ELECTION_CYCLE_FILENAME:
    raise ValueError("Election cycle filename not found. Check env")
if not INSIGHTS_GENERALISED_PROMPT:
    raise ValueError("Insights prompt not found. Check env")
if not GENERAL_STRATEGY_PROMPT:
    raise ValueError("General prompt not found. Check env")
if not KB_ID:
    raise ValueError("Knowledge base ID not found. Check env")

logger = helpers._setup_logger(name='chatbot', level=20)
CANDIDATE_CONTEXT = {
    "office_position": "House_of_Delegates",
    "district_name": "District_41",
    "current_year": 2025
}
with open(ELECTION_CYCLE_FILENAME, 'r') as f:
    ELECTION_CYCLES_DATA = json.loads(f.read())

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name='us-east-1'
)

config = Config(
    read_timeout=300,
    connect_timeout=60,
    retries={
        'total_max_attempts': 5,
        'mode': 'adaptive'
    }
)

class PCMChatbot():
    def __init__(self,
                 scope_model_id: str,
                 model_id: str,
                 s3_client: Any,
                 bedrock_runtime: Any,
                 bedrock_agent_runtime: Any,
                 candidate_context: Dict[str, Any]):
        self.scope_model_id = scope_model_id
        self.model_id = model_id
        self.s3_client = s3_client
        self.bedrock_runtime = bedrock_runtime
        self.bedrock_agent_runtime = bedrock_agent_runtime
        self.candidate_context = candidate_context

    def load_data(self):
        # Create S3 paths for retrieval of election data.
        # logger.info("\x1b[33mCreating S3 paths of candidates election data and other office's election data\x1b[0m")
        print("\x1b[33mCreating S3 paths of candidates election data and other office's election data\x1b[0m")
        retrieval_plan = self.generate_retrieval_plan_from_election_cycles()
        # Load data from S3.
        # logger.info("\x1b[33mExtracting data from S3\x1b[0m")
        spinner = utils.Spinner("\x1b[33mExtracting data from S3\x1b[0m")
        spinner.start()
        # print("\x1b[33mExtracting data from S3\x1b[0m")
        try:
            candidate_election_data, other_election_data = self.extract_data_from_s3_all_elections(retrieval_plan=retrieval_plan,
                                                                    district_name=self.candidate_context['district_name'])
        finally:
            spinner.stop()
            print("\x1b[32mDone\x1b[0m")
        
        return candidate_election_data, other_election_data

    def generate_retrieval_plan_from_election_cycles(
        self
    ) -> Dict[str, Any]:
        """
        Generate the retrieval plan automatically based on candidate context and election cycles.
        
        Args:
            candidate_context: Dict with office_position, district_name, current_year
            scope: "ALL_ELECTIONS" or "CANDIDATE_OFFICE_ONLY" or "None"
        
        Returns:
            Retrieval plan in the same format as before
        """
        candidate_office = self.candidate_context['office_position']
        district_name = self.candidate_context['district_name']
        current_year = self.candidate_context['current_year']
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
                continue  # Skip candidate's office (already added)
            
            # Get election years for this office in the lookback window
            office_election_years = self._get_election_years_in_window(office_name, current_year, lookback_years)
            
            # Add to retrieval plan
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
        
        # Calculate totals
        total_elections = len(retrieval_plan['candidate_office_data']) + len(retrieval_plan['all_other_elections'])
        retrieval_plan['analysis_context']['total_elections_retrieved'] = total_elections
        
        return retrieval_plan
    @helpers.measure_execution_time
    def extract_data_from_s3_all_elections(self, retrieval_plan: Dict[str, Any], district_name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract precinct-level data from S3 for all elections.
        For statewide races, filters to only precincts that exist in the candidate's district.
        """
        bucket_name = "predictif-election-data"
        
        # First pass: Extract candidate office data to get the list of precincts
        candidate_office_data = []
        for plan_item in retrieval_plan.get('candidate_office_data', []):
            s3_path = plan_item['s3_path']
            extraction_spec = plan_item['extraction_spec']
            election_info = plan_item.get('election_info', {})
            purpose = plan_item.get('purpose', '')
            
            try:
                response = self.s3_client.get_object(Bucket=bucket_name, Key=s3_path)
                full_data = json.loads(response['Body'].read().decode('utf-8'))
                
                # Extract district-specific precinct data
                extracted_chunk = self._extract_district_precinct_detail(full_data, district_name)
                
                # Add metadata
                extracted_chunk['_metadata'] = {
                    'source_path': s3_path,
                    'category': 'candidate_office',
                    'purpose': purpose,
                    'election_info': election_info,
                    'extraction_level': 'DISTRICT_PRECINCT_DETAIL'
                }
                
                candidate_office_data.append(extracted_chunk)
                logger.debug(f"Extracted data from: {s3_path}")
                
            except Exception as e:
                logger.debug(f"Error extracting candidate office data from {s3_path}: {str(e)}")
        
        # Get the set of precinct names from candidate's district
        candidate_precincts = self._get_candidate_district_precincts(candidate_office_data)
        other_election_data = []
        # Second pass: Extract all other elections data
        for plan_item in retrieval_plan.get('all_other_elections', []):
            s3_path = plan_item['s3_path']
            extraction_spec = plan_item['extraction_spec']
            election_info = plan_item.get('election_info', {})
            purpose = plan_item.get('purpose', '')
            is_statewide = election_info.get('is_statewide', False)
            
            try:
                response = self.s3_client.get_object(Bucket=bucket_name, Key=s3_path)
                full_data = json.loads(response['Body'].read().decode('utf-8'))
                
                # Get the district name from filters
                filter_district = extraction_spec.get('filters', {}).get('district_name', district_name)
                
                if is_statewide:
                    # For statewide races, extract and filter to candidate's precincts
                    extracted_chunk = self._extract_statewide_filtered_to_district(
                        full_data,
                        filter_district,
                        candidate_precincts
                    )
                else:
                    # For district-based races, extract normally
                    extracted_chunk = self._extract_district_precinct_detail(full_data, filter_district)
                
                # Add metadata
                extracted_chunk['_metadata'] = {
                    'source_path': s3_path,
                    'category': 'other_elections',
                    'purpose': purpose,
                    'election_info': election_info,
                    'extraction_level': 'DISTRICT_PRECINCT_DETAIL',
                    'is_statewide': is_statewide
                }
                
                other_election_data.append(extracted_chunk)
                logger.debug(f"Extracted data from: {s3_path}")
                
            except Exception as e:
                logger.debug(f"Error extracting other election data from {s3_path}: {str(e)}")
        
        return candidate_office_data, other_election_data
    
    # Understand what kind of question the user has asked---------------------
    def get_scope_decision_from_bedrock(self,
                                        user_query: str,
                                        candidate_context: Dict[str, Any],
                                        model_id: str,
                                        prompt_path: str='texts/scope_decision_prompt.txt') -> str:
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
        with open(prompt_path, 'r') as f:
            system_prompt = f.read()
        
        # Generate election cycles context for LLM
        election_cycles_context = self._generate_election_cycles_context(5)  # Last 5 years
        
        # Replace placeholders
        full_prompt = system_prompt.replace('{{OFFICE_POSITION}}', candidate_context['office_position'])
        full_prompt = full_prompt.replace('{{DISTRICT_NAME}}', candidate_context['district_name'])
        full_prompt = full_prompt.replace('{{CURRENT_YEAR}}', str(candidate_context['current_year']))
        full_prompt = full_prompt.replace('{{USER_QUERY}}', user_query)
        full_prompt = full_prompt.replace('{{ELECTION_CYCLES}}', election_cycles_context)
        
        # Call Bedrock
        response = self.bedrock_runtime.converse(
            modelId=model_id,
            messages=[
                {
                    'role': 'user',
                    'content': [{'text': full_prompt}]
                }
            ],
            inferenceConfig={
                'temperature': 0.1,
                'maxTokens': 500  # Small output, just JSON
            }
        )
        
        # Parse response
        response_text = response['output']['message']['content'][0]['text']
        
        try:
            decision = json.loads(response_text)
            scope = decision.get('scope', 'ALL_ELECTIONS')  # Default to ALL_ELECTIONS
            logger.info(f"Scope decision: {scope}")
            logger.info(f"Reasoning: {decision.get('reasoning', 'N/A')}")
            return scope
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(1))
                return decision.get('scope', 'ALL_ELECTIONS')
            else:
                logger.warning("Could not parse scope decision, defaulting to ALL_ELECTIONS")
                return 'ALL_ELECTIONS'
            
    def create_llm_prompt_with_context_all_elections(self,
                                                     user_query: str, 
                                                     extracted_data: Optional[List[Dict[str, Any]]], 
                                                     candidate_context: Dict[str, Any]) -> str:
        """
        Create the final prompt for the LLM with all elections precinct-level context.
        """
        election_laws = self._retrieve_laws(user_query=user_query)
        if election_laws:
            election_laws = f"RELEVANT ELECTION LAWS:\n-----------------------\n{election_laws}"
        if extracted_data:
            context = self._format_for_llm_context_all_elections(extracted_data, candidate_context)
            with open("texts/election_data_context.txt", 'w') as f:
                f.write(context)
            election_cycles_context = self._generate_election_cycles_context(5)
            with open(os.path.join(PROMPT_FOLDER, INSIGHTS_GENERALISED_PROMPT), 'r') as f:
                prompt = f.read()

            prompt = prompt.format(office_position=candidate_context.get('office_position'),
                                   district_name=candidate_context.get('district_name'),
                                   current_year=candidate_context.get('current_year'),
                                   user_query=user_query,
                                   election_cycles_context=election_cycles_context,
                                   election_laws=election_laws,
                                   context=context)
        else:
            with open(os.path.join(PROMPT_FOLDER, GENERAL_STRATEGY_PROMPT, 'r')) as f:
                prompt = f.read()
            
            prompt = prompt.format(office_position=candidate_context.get('office_position'),
                                   district_name=candidate_context.get('district_name'),
                                   current_year=candidate_context.get('current_year'),
                                   election_laws=election_laws,
                                   user_query=user_query)

        return prompt
    
    @helpers.measure_execution_time
    def get_answer_from_bedrock(self, conversation: List, prompt: str) -> Dict:
        """
        Call Bedrock to get the final strategic answer.
        """
        message = {
                    'role': 'user',
                    'content': [{'text': prompt}]
                }
        messages = conversation + [message]
        response = self.bedrock_runtime.converse(
            modelId=self.model_id,
            messages=messages,
            inferenceConfig={
                'temperature': 0.3,
                'maxTokens': 8000
            }
        )
    
        return response
    
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
    
    def _format_for_llm_context_all_elections(self, extracted_data: List[Dict[str, Any]], candidate_context: Dict[str, Any]) -> str:
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
            
            # Header
            year = data.get('year', 'N/A')
            stage = data.get('stage', 'N/A')
            office = election_info.get('office', data.get('office', 'Unknown'))
            is_pres_year = " (PRESIDENTIAL YEAR)" if election_info.get('is_presidential_year') else ""
            statewide_note = " [Filtered to District Precincts]" if is_statewide else ""
            
            parts.append("\n" + "─" * 80)
            parts.append(f"📊 {office} {year} {stage}{is_pres_year}{statewide_note}")
            parts.append("─" * 80)
            parts.append(f"Purpose: {metadata.get('purpose', 'N/A')}")
            
            # # Check for errors
            # if 'error' in metadata:
            #     parts.append(f"⚠️  ERROR: {metadata['error']}")
            #     return "\n".join(parts)
            
            # if '_error' in data:
            #     parts.append(f"⚠️  {data['_error']}")
            #     return "\n".join(parts)
            
            # if '_warning' in data:
            #     parts.append(f"⚠️  WARNING: {data['_warning']}")
            
            district = data.get('district')
            if not district:
                parts.append("⚠️  No district data available")
                return "\n".join(parts)
            
            parts.append("")
            parts.append(f"District: {district.get('district_name', 'Unknown')}")
            parts.append(f"  📍 District Total Votes: {district.get('district_total_votes', 0):,}")
            
            if is_statewide:
                parts.append(f"  ℹ️  Note: Statewide race filtered to show only precincts in your district")
            
            # Show win/flip numbers if available (not null)
            if district.get('district_win_number') is not None:
                parts.append(f"  🎯 Votes Needed to Win: {district.get('district_win_number'):,}")
            
            if district.get('district_flip_number') is not None:
                parts.append(f"  🔄 Votes Needed to Flip: {district.get('district_flip_number'):,}")
            
            parts.append("")
            parts.append("PRECINCT BREAKDOWN:")
            parts.append("")
            
            # Get precincts
            precincts = district.get('precincts', [])
            
            # Sort precincts by total votes (highest first)
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
                
                # Candidate results
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
                    
                    # Calculate margin if there are at least 2 candidates
                    if len(results_sorted) >= 2:
                        margin = results_sorted[0].get('votes', 0) - results_sorted[1].get('votes', 0)
                        margin_pct = (margin / precinct_total * 100) if precinct_total > 0 else 0
                        parts.append(f"     Margin: {margin:,} votes ({margin_pct:.1f}%)")
                
                parts.append("")
            
            # District-level summary
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
        context_parts.append(f"Candidate Information:")
        context_parts.append(f"  Office: {candidate_context.get('office_position', 'Unknown')}")
        context_parts.append(f"  District: {candidate_context.get('district_name', 'Unknown')}")
        context_parts.append(f"  Analysis Year: {candidate_context.get('current_year', 'Unknown')}")
        context_parts.append("")
        context_parts.append("DATA SCOPE:")
        context_parts.append("  • 5 years of historical precinct-level data")
        context_parts.append("  • All elections that occurred in your district during this period")
        context_parts.append("  • Includes primaries and general elections for all offices")
        context_parts.append("  • For statewide races: ONLY precincts in your district")
        context_parts.append("")
        context_parts.append("=" * 80)
        context_parts.append("")
        
        # Separate candidate office data from other elections
        candidate_data = [d for d in extracted_data if d.get('_metadata', {}).get('category') == 'candidate_office']
        other_data = [d for d in extracted_data if d.get('_metadata', {}).get('category') == 'other_elections']
        
        # Sort by year (most recent first)
        candidate_data.sort(key=lambda x: x.get('year', 0), reverse=True)
        other_data.sort(key=lambda x: x.get('year', 0), reverse=True)
        
        # Format candidate office data
        if candidate_data:
            context_parts.append(f"YOUR OFFICE ELECTION HISTORY ({candidate_context.get('office_position', 'Unknown')}):")
            context_parts.append("=" * 80)
            context_parts.append("Precinct-level data for your office over the past 5 years")
            context_parts.append("")
            for data in candidate_data:
                context_parts.append(_format_precinct_chunk(data))
            context_parts.append("")
        
        # Format other elections data
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
        current_year = datetime.datetime.now().year
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
                # Even-numbered years (President, U.S. House, etc.)
                elections_in_window = [y for y in years if y % 2 == 0]
            elif pattern == 'odd':
                # Odd-numbered years (Governor, House_of_Delegates, etc.)
                elections_in_window = [y for y in years if y % 2 == 1]
            elif pattern == 'even_biennial':
                # Senate: every 2 years in even years (but not all states every 2 years, some every 6)
                elections_in_window = [y for y in years if y % 2 == 0]
            elif pattern == 'annual':
                # Annual elections
                elections_in_window = years
            elif pattern == 'periodic':
                # Periodic elections (every 5 years)
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
        
        # Find the election definition
        election_def = None
        for election in ELECTION_CYCLES_DATA['elections']:
            if election['election'] == office_name:
                election_def = election
                break
        
        if not election_def:
            logger.warning(f"Office {office_name} not found in election cycles")
            return []
        
        cycle = election_def['cycle']
        pattern = election_def['election_pattern']
        
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
        
        logger.debug(f"Election years for {office_name}: {elections_in_window}")
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
        
        logger.debug(f"Found {len(precinct_names)} unique precincts in candidate's district")
        # logger.debug(f"Precinct names: {precinct_names}")
        
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
        
        # Find the specific district
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
        
        # Find the statewide district
        statewide_district = None
        for district in data.get('districts', []):
            if district.get('district_name') == district_name:
                statewide_district = district
                break
        
        if statewide_district is None:
            result['_error'] = f"District {district_name} not found in data"
            return result
        
        # Filter precincts to only those in the candidate's district
        filtered_precincts = []
        all_precincts = statewide_district.get('precincts', [])
        
        for precinct in all_precincts:
            precinct_name = precinct.get('precinct_name')
            if precinct_name in candidate_precincts:
                filtered_precincts.append(precinct)
        
        logger.debug(f"Filtered statewide {data.get('office')} {data.get('year')} {data.get('stage')}: "
                    f"{len(all_precincts)} total precincts -> {len(filtered_precincts)} matching candidate's district")
        
        # Create filtered district
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

def main(scope_model_id: str):
    extracted_data = None
    bedrock_runtime = session.client('bedrock-runtime', config=config)
    s3_client = session.client("s3")
    bedrock_agent_runtime = session.client("bedrock-agent-runtime")
    chatbot = PCMChatbot(scope_model_id=scope_model_id,
                         model_id=scope_model_id,
                         s3_client=s3_client,
                         bedrock_runtime=bedrock_runtime,
                         bedrock_agent_runtime=bedrock_agent_runtime,
                         candidate_context=CANDIDATE_CONTEXT)
    # Load candidates election data and other election's data for use.
    candidate_election_data, other_election_data = chatbot.load_data()
    print("Hello, I'm your personal Political Campaign manager, how can I help you today?")
    conversation = []
    while True:
        user_query = input("\x1b[32m***> \x1b[0m")
        if user_query.lower() == 'n':
            break
        # logger.info("\x1b[33mGetting scope decision from Bedrock...\x1b[0m")
        print("\x1b[33mGetting scope decision from Bedrock...\x1b[0m")
        # Step 1: Ask an LLM to figure out what kind of question the user has asked.
        scope = chatbot.get_scope_decision_from_bedrock(user_query=user_query,
                                                        candidate_context=CANDIDATE_CONTEXT,
                                                        model_id=scope_model_id)
        # logger.info("\x1b[33mAppending relevant data based on scope...\x1b[0m")
        print("\x1b[33mAppending relevant data based on scope...\x1b[0m")
        if scope == 'ALL_ELECTIONS':
            extracted_data = candidate_election_data + other_election_data
        elif scope == 'CANDIDATE_OFFICE_ONLY':
            extracted_data = candidate_election_data
        else:
            extracted_data = None

        # logger.info("\x1b[33mFormatting data for LLM...\x1b[0m")
        print("\x1b[33mFormatting data for LLM...\x1b[0m")
        textual_data = chatbot.create_llm_prompt_with_context_all_elections(user_query=user_query,
                                                                extracted_data=extracted_data,
                                                                candidate_context=CANDIDATE_CONTEXT)
        with open('texts/final_prompt.txt', 'w') as f:
            f.write(textual_data)
        # logger.info("\x1b[33mFinal response from Bedrock...\x1b[0m")
        print("\x1b[33mFinal response from Bedrock...\x1b[0m")
        response = chatbot.get_answer_from_bedrock(conversation=conversation, prompt=textual_data)
        conversation_user_message = {
                    'role': 'user',
                    'content': [{'text': user_query}]
                }
        conversation.append(conversation_user_message)
        conversation_ai_message = response['output']['message']
        conversation.append(conversation_ai_message)
        answer = response['output']['message']['content'][0]['text']
        print(answer)
        control = input("\n\nDo you wish to continue? \x1b[32mY\x1b[0m \\ \x1b[31mN\x1b[0m: ")
        if control.lower() == 'n':
            break


if __name__ == "__main__":
    scope_model_id = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
    main(scope_model_id=scope_model_id)