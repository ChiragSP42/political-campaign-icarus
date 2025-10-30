#%%

import json
import boto3
from botocore.config import Config
from aws_helpers import helpers
import os
import datetime
import election_context
from dotenv import load_dotenv
from typing import Dict, List, Any, Set

load_dotenv(override=True)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", None)
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", None)
logger = helpers._setup_logger(name='election_retrieval', level=10)

if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("AWS credentials not found")

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name='us-east-1'
)

config = Config(
    read_timeout=300,      # 5 minutes for read operations
    connect_timeout=60,    # 1 minute for initial connection
    retries={
        'total_max_attempts': 5,    # Total attempts including initial request
        'mode': 'adaptive'          # Adaptive retry mode for intelligent backoff
    }
)

bedrock_runtime = session.client('bedrock-runtime', config=config)
s3_client = session.client("s3")


def get_retrieval_plan_from_bedrock(user_query: str, candidate_context: Dict[str, Any], model_id: str) -> Dict:
    """
    Call Bedrock to get the retrieval plan for all elections at precinct level.
    """
    with open('texts/content_retrieval_S3_data_chunk_prompt.txt', 'r') as f:
        system_prompt = f.read()
    
    # Replace placeholders
    full_prompt = system_prompt.replace('{{OFFICE_POSITION}}', candidate_context['office_position'])
    full_prompt = full_prompt.replace('{{DISTRICT_NAME}}', candidate_context['district_name'])
    full_prompt = full_prompt.replace('{{CURRENT_YEAR}}', str(candidate_context['current_year']))
    full_prompt = full_prompt.replace('{{USER_QUERY}}', user_query)

    # election_cycles = election_context.load_office_mapping_from_file(file_path='election_cycle_testing.json')
    # current_date = datetime.datetime.now()
    # llm_context = election_context.generate_llm_context(election_cycles, current_date, lookback_years=5)

    # full_prompt = full_prompt.replace('{{ELECTION_CYCLES}}', llm_context)
    
    # Call Bedrock
    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=[
            {
                'role': 'user',
                'content': [{'text': full_prompt}]
            }
        ],
        inferenceConfig={
            'temperature': 0.1,
            'maxTokens': 8000
        }
    )
    
    # Parse response
    response_text = response['output']['message']['content'][0]['text']
    while True:
        try:
            retrieval_plan = json.loads(response_text)
            return retrieval_plan
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                retrieval_plan = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    retrieval_plan = json.loads(json_match.group(0))
                else:
                    raise ValueError("Could not parse JSON from Bedrock response")
        


def get_candidate_district_precincts(extracted_candidate_data: List[Dict[str, Any]]) -> Set[str]:
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
    
    logger.info(f"Found {len(precinct_names)} unique precincts in candidate's district")
    logger.debug(f"Precinct names: {precinct_names}")
    
    return precinct_names


def extract_data_from_s3_all_elections(retrieval_plan: Dict[str, Any], district_name: str) -> List[Dict[str, Any]]:
    """
    Extract precinct-level data from S3 for all elections.
    For statewide races, filters to only precincts that exist in the candidate's district.
    """
    extracted_data = []
    bucket_name = "predictif-election-data"
    
    # First pass: Extract candidate office data to get the list of precincts
    candidate_office_data = []
    for plan_item in retrieval_plan.get('candidate_office_data', []):
        s3_path = plan_item['s3_path']
        extraction_spec = plan_item['extraction_spec']
        election_info = plan_item.get('election_info', {})
        purpose = plan_item.get('purpose', '')
        
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_path)
            full_data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Extract district-specific precinct data
            extracted_chunk = extract_district_precinct_detail(full_data, district_name)
            
            # Add metadata
            extracted_chunk['_metadata'] = {
                'source_path': s3_path,
                'category': 'candidate_office',
                'purpose': purpose,
                'election_info': election_info,
                'extraction_level': 'DISTRICT_PRECINCT_DETAIL'
            }
            
            candidate_office_data.append(extracted_chunk)
            extracted_data.append(extracted_chunk)
            logger.info(f"Extracted data from: {s3_path}")
            
        except Exception as e:
            logger.error(f"Error extracting candidate office data from {s3_path}: {str(e)}")
            # extracted_data.append({
            #     '_metadata': {
            #         'source_path': s3_path,
            #         'category': 'candidate_office',
            #         'purpose': purpose,
            #         'error': str(e)
            #     }
            # })
    
    # Get the set of precinct names from candidate's district
    candidate_precincts = get_candidate_district_precincts(candidate_office_data)
    
    # Second pass: Extract all other elections data
    for plan_item in retrieval_plan.get('all_other_elections', []):
        s3_path = plan_item['s3_path']
        extraction_spec = plan_item['extraction_spec']
        election_info = plan_item.get('election_info', {})
        purpose = plan_item.get('purpose', '')
        is_statewide = election_info.get('is_statewide', False)
        
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_path)
            full_data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Get the district name from filters
            filter_district = extraction_spec.get('filters', {}).get('district_name', district_name)
            
            if is_statewide:
                # For statewide races, extract and filter to candidate's precincts
                extracted_chunk = extract_statewide_filtered_to_district(
                    full_data, 
                    filter_district, 
                    candidate_precincts
                )
            else:
                # For district-based races, extract normally
                extracted_chunk = extract_district_precinct_detail(full_data, filter_district)
            
            # Add metadata
            extracted_chunk['_metadata'] = {
                'source_path': s3_path,
                'category': 'other_elections',
                'purpose': purpose,
                'election_info': election_info,
                'extraction_level': 'DISTRICT_PRECINCT_DETAIL',
                'is_statewide': is_statewide
            }
            
            extracted_data.append(extracted_chunk)
            
        except Exception as e:
            logger.error(f"Error extracting other election data from {s3_path}: {str(e)}")
            extracted_data.append({
                '_metadata': {
                    'source_path': s3_path,
                    'category': 'other_elections',
                    'purpose': purpose,
                    'error': str(e)
                }
            })
    
    return extracted_data


def extract_district_precinct_detail(data: Dict, district_name: str) -> Dict:
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


def extract_statewide_filtered_to_district(data: Dict, district_name: str, candidate_precincts: Set[str]) -> Dict:
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
    
    logger.info(f"Filtered statewide {data.get('office')} {data.get('year')} {data.get('stage')}: "
                f"{len(all_precincts)} total precincts -> {len(filtered_precincts)} matching candidate's district")
    
    # Create filtered district
    filtered_district = {
        'district_name': district_name,
        'district_total_votes': sum(p.get('precinct_total_votes', 0) for p in filtered_precincts),
        'district_win_number': None,  # Not applicable for filtered statewide data
        'district_flip_number': None,
        'precincts': filtered_precincts
    }
    
    result['district'] = filtered_district
    
    if not filtered_precincts:
        result['_warning'] = f"No matching precincts found in statewide data for candidate's district"
    
    return result


def format_for_llm_context_all_elections(extracted_data: List[Dict[str, Any]], candidate_context: Dict[str, Any]) -> str:
    """
    Format extracted election data for LLM consumption - all precinct level.
    """
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
    context_parts.append("  • For statewide races (Governor, President): ONLY precincts in your district")
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
            context_parts.append(format_precinct_chunk(data))
        context_parts.append("")
    
    # Format other elections data
    if other_data:
        context_parts.append("ALL OTHER ELECTIONS IN YOUR DISTRICT:")
        context_parts.append("=" * 80)
        context_parts.append("Precinct-level data for all other offices - SAME PRECINCTS as your district")
        context_parts.append("")
        for data in other_data:
            context_parts.append(format_precinct_chunk(data))
        context_parts.append("")
    
    return "\n".join(context_parts)


def format_precinct_chunk(data: Dict[str, Any]) -> str:
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
    
    # Check for errors
    if 'error' in metadata:
        parts.append(f"⚠️  ERROR: {metadata['error']}")
        return "\n".join(parts)
    
    if '_error' in data:
        parts.append(f"⚠️  {data['_error']}")
        return "\n".join(parts)
    
    if '_warning' in data:
        parts.append(f"⚠️  WARNING: {data['_warning']}")
    
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


def create_llm_prompt_with_context_all_elections(user_query: str, extracted_data: List[Dict[str, Any]], candidate_context: Dict[str, Any]) -> str:
    """
    Create the final prompt for the LLM with all elections precinct-level context.
    """
    context = format_for_llm_context_all_elections(extracted_data, candidate_context)
    # election_cycles = election_context.load_office_mapping_from_file(file_path='election_cycle_testing.json')
    # current_date = datetime.datetime.now()
    # llm_context = election_context.generate_llm_context(election_cycles, current_date, lookback_years=5)
    
    prompt = f"""CANDIDATE PROFILE:
Office: {candidate_context.get('office_position')} | District: {candidate_context.get('district_name')} | Year: {candidate_context.get('current_year')}

QUESTION: {user_query}

ELECTION DATA (includes precalculated Win/Flip Numbers): {context}

INSTRUCTIONS:
You are a specialized election strategist. Provide a data-driven strategic answer using the precalculated Win and Flip Numbers in the election data. Focus on actionable insights at both district and precinct levels.

## REQUIRED OUTPUTS:

### 1. EXECUTIVE SUMMARY (Start with this)
- **Win Number**: [X] votes needed to win (from data)
- **Flip Number**: [X] votes needed to flip district (from data)
- **Current Situation**: [Last result, current gap]
- **Recommended Path**: [One sentence with specific numbers]

### 2. TURNOUT SCENARIO ANALYSIS
Present Win/Flip Numbers for all turnout scenarios provided in the data:

| Scenario | Expected Turnout | Win Number | Flip Number | Win % | Strategy Recommendation |
|----------|------------------|------------|-------------|-------|------------------------|
| Base     | [X] votes        | [Y] votes  | [Z] votes   | [%]   | [Brief strategy]       |
| Realistic| [X] votes        | [Y] votes  | [Z] votes   | [%]   | [Brief strategy]       |
| High     | [X] votes        | [Y] votes  | [Z] votes   | [%]   | [Brief strategy]       |

**Recommended Scenario**: [Choose and explain why based on historical patterns]

### 3. PRECINCT TARGETING TABLE
Using data provided, categorize and prioritize every precinct:

| Precinct Name | Expected Turnout | Partisan Lean (%) | Category | Net Votes Toward Win | Priority | Strategy |
|---------------|------------------|-------------------|----------|----------------------|----------|----------|
| [All precincts from data] |

**Categories**:
- **FLIP**: Opposition precinct (<-5% lean) - votes needed to flip
- **DEFEND**: Your precinct at risk (+5 to +15% lean) - votes to maintain
- **MAXIMIZE**: Strong base (>+15% lean) - additional votes via turnout
- **PERSUADE**: True swing (-5 to +5% lean) - persuadable voters

**Priority Tiers**:
- Tier 1: High impact, achievable targets
- Tier 2: Moderate impact or difficulty
- Tier 3: Aspirational or defensive holds

### 4. STRATEGIC PATHWAYS
Show 2-3 specific scenarios demonstrating how to reach Win Number:

**Path A: [Strategy Name]**
- Target precincts: [List with specific names]
- Actions: [Specific tactics]
- Vote impact: +[X] votes
- Result: [Win margin]

**Path B: [Strategy Name]**
- Target precincts: [List with specific names]
- Actions: [Specific tactics]
- Vote impact: +[X] votes
- Result: [Win margin]

**Recommended Path**: [Choose and explain]

### 5. CROSS-ELECTION INSIGHTS
Analyze voter behavior across different races in SAME precincts:
- **Ticket-Splitters**: [X] voters, [Y] precincts - persuasion targets
- **Drop-Off Voters**: [X] voters who skipped this race - mobilization targets
- **Surge Voters**: [X] additional voters in presidential years - off-year risk/opportunity
- **Partisan Shifts**: Precincts trending toward/away from your party

### 6. RESOURCE ALLOCATION
Based on precinct priorities:
- **Tier 1 Precincts**: [X]% of resources → [Y] expected votes
- **Tier 2 Precincts**: [X]% of resources → [Y] expected votes
- **Tier 3 Precincts**: [X]% of resources → [Y] expected votes
- **Expected Total Gain**: [X] votes toward Win Number

### 7. KEY RECOMMENDATIONS
- **Primary Targets**: [Specific precincts] = [X] votes needed
- **Turnout Goals**: [Precinct types] from [current %] to [target %]
- **Persuasion Targets**: [X] voters in [Y] swing precincts
- **Defense Strategy**: Maintain [X] votes across [Y] at-risk precincts
- **Cumulative Impact**: [Show how actions sum to Win Number]

## ANALYSIS REQUIREMENTS:
- Use Win/Flip Numbers directly from provided data - DO NOT recalculate
- Every statement must reference specific precincts and vote counts from data
- Show how precinct-level actions aggregate to district Win Number
- Analyze cross-election patterns (same precincts across different races)
- Prioritize based on efficiency: votes gained per resource invested
- Be specific and actionable - avoid general observations
- Use tables extensively for clarity
- Bold all critical numbers: **[X] votes**

## OUTPUT FORMAT:
- Start immediately with Executive Summary - no preamble
- Use clear section headers and tables
- Present actual precinct names and numbers from data
- End with concise Bottom Line summarizing the path to victory

Provide strategic analysis with specific vote targets, named precincts, and clear mathematical path to reaching the Win Number.
"""
    
    return prompt


def get_answer_from_bedrock(prompt: str, model_id: str) -> str:
    """
    Call Bedrock to get the final strategic answer.
    """
    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=[
            {
                'role': 'user',
                'content': [{'text': prompt}]
            }
        ],
        inferenceConfig={
            'temperature': 0.3,
            'maxTokens': 8000
        }
    )
    
    return response['output']['message']['content'][0]['text']


def main(user_query, candidate_context, model_id):
    """
    Main function to orchestrate the retrieval and analysis.
    
    Args:
        user_query: The candidate's question
        candidate_context: Dict with office_position, district_name, current_year
        model_id: Bedrock model ID to use
    """
    # Step 1: Get retrieval plan from Bedrock
    logger.info("Step 1: Getting retrieval plan from Bedrock...")
    retrieval_plan = get_retrieval_plan_from_bedrock(user_query, candidate_context=candidate_context, model_id=model_id)
    # logger.debug(json.dumps(retrieval_plan, indent=2))
    
    # Step 2: Extract data from S3
    logger.info("Step 2: Extracting data from S3...")
    extracted_data = extract_data_from_s3_all_elections(
        retrieval_plan,
        candidate_context['district_name']
    )
    
    # Step 3: Format for LLM
    logger.info("Step 3: Formatting data for LLM...")
    final_prompt = create_llm_prompt_with_context_all_elections(
        user_query,
        extracted_data,
        candidate_context
    )
    
    with open("texts/election_data_context.txt", 'w') as f:
        f.write(final_prompt)
    
    # Step 4: Get final answer
    logger.info("Step 4: Getting final answer from Bedrock...")
    answer = get_answer_from_bedrock(final_prompt, model_id=model_id)
    
    return {
        'statusCode': 200,
        'body': {
            'answer': answer,
            'retrieval_plan': retrieval_plan,
            'data_sources': [d['_metadata']['source_path'] for d in extracted_data if '_metadata' in d]
        }
    }


if __name__ == "__main__":
    MODEL_ID = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
    # MODEL_ID = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
    # MODEL_ID = 'us.anthropic.claude-3-haiku-20240307-v1:0'
    
    # Example: House of Delegates candidate
    candidate_context_hod = {
        "office_position": "House_of_Delegates",
        "district_name": "District_3",
        "current_year": 2025
    }
    user_query_hod = 'How can I, a democratic candidate, win the election?'
    
    # Run the example
    final_response = main(user_query=user_query_hod, candidate_context=candidate_context_hod, model_id=MODEL_ID)
    
    logger.info(f"\n\n{final_response['body']['answer']}")
# %%
