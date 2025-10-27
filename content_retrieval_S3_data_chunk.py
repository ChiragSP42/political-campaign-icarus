#%%
import json
import boto3
from aws_helpers import helpers
import os
from dotenv import load_dotenv
from typing import Dict, List, Any

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

bedrock_runtime = session.client('bedrock-runtime')
s3_client = session.client("s3")


def get_retrieval_plan_from_bedrock(user_query: str, candidate_context: Dict[str, Any], model_id: str) -> Dict:
    """
    Call Bedrock to get the district-focused retrieval plan.
    """
    with open('content_retrieval_S3_data_chunk_prompt.txt', 'r') as f:
        system_prompt = f.read()
    
    # Replace placeholders
    full_prompt = system_prompt.replace('{{OFFICE_POSITION}}', candidate_context['office_position'])
    full_prompt = full_prompt.replace('{{DISTRICT_NAME}}', candidate_context['district_name'])
    full_prompt = full_prompt.replace('{{CURRENT_YEAR}}', str(candidate_context['current_year']))
    full_prompt = full_prompt.replace('{{USER_QUERY}}', user_query)
    
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
    
    try:
        retrieval_plan = json.loads(response_text)
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
    
    return retrieval_plan


def extract_data_from_s3_district_focused(retrieval_plan: Dict[str, Any], district_name: str, office_position: str) -> List[Dict[str, Any]]:
    """
    Extract data from S3 based on district-focused retrieval plan.
    Handles both district-based races (House/Senate) and statewide races (Governor/President).
    """
    extracted_data = []
    bucket_name = "predictif-election-data"
    
    # Determine if this is a statewide race
    is_statewide = district_name == "Statewide"
    
    # Extract candidate district data
    for plan_item in retrieval_plan.get('candidate_district_data', []):
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
                'category': 'candidate_district',
                'purpose': purpose,
                'election_info': election_info,
                'extraction_level': 'DISTRICT_PRECINCT_DETAIL',
                'is_statewide': is_statewide
            }
            
            extracted_data.append(extracted_chunk)
            
        except Exception as e:
            logger.error(f"Error extracting candidate data from {s3_path}: {str(e)}")
            extracted_data.append({
                '_metadata': {
                    'source_path': s3_path,
                    'category': 'candidate_district',
                    'purpose': purpose,
                    'error': str(e)
                }
            })
    
    # Extract contextual election data
    for plan_item in retrieval_plan.get('contextual_elections', []):
        s3_path = plan_item['s3_path']
        extraction_spec = plan_item['extraction_spec']
        election_info = plan_item.get('election_info', {})
        purpose = plan_item.get('purpose', '')
        
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_path)
            full_data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Extract based on level
            level = extraction_spec.get('level', 'FILE_LEVEL')
            if level == 'FILE_LEVEL':
                extracted_chunk = extract_file_level(full_data)
            elif level == 'DISTRICT_SUMMARY':
                extracted_chunk = extract_district_summary(full_data)
            elif level == 'DISTRICT_PRECINCT_DETAIL':
                # For contextual data, might also want district detail
                filter_district = extraction_spec.get('filters', {}).get('district_name', 'Statewide')
                extracted_chunk = extract_district_precinct_detail(full_data, filter_district)
            else:
                extracted_chunk = extract_file_level(full_data)
            
            # Add metadata
            extracted_chunk['_metadata'] = {
                'source_path': s3_path,
                'category': 'contextual',
                'purpose': purpose,
                'election_info': election_info,
                'extraction_level': level
            }
            
            extracted_data.append(extracted_chunk)
            
        except Exception as e:
            logger.error(f"Error extracting contextual data from {s3_path}: {str(e)}")
            extracted_data.append({
                '_metadata': {
                    'source_path': s3_path,
                    'category': 'contextual',
                    'purpose': purpose,
                    'error': str(e)
                }
            })
    
    return extracted_data


def extract_district_precinct_detail(data: Dict, district_name: str) -> Dict:
    """
    Extract full precinct-level detail for a specific district.
    Works for both district-based races (District_41) and statewide races (Statewide).
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


def extract_file_level(data: Dict) -> Dict:
    """Extract only top-level metadata."""
    return {
        'record_id': data.get('record_id'),
        'office': data.get('office'),
        'year': data.get('year'),
        'stage': data.get('stage'),
        'total_votes': data.get('total_votes')
    }


def extract_district_summary(data: Dict) -> Dict:
    """Extract district-level aggregates without precinct details."""
    result = {
        'record_id': data.get('record_id'),
        'office': data.get('office'),
        'year': data.get('year'),
        'stage': data.get('stage'),
        'total_votes': data.get('total_votes'),
        'districts': []
    }
    
    for district in data.get('districts', []):
        result['districts'].append({
            'district_name': district.get('district_name'),
            'district_total_votes': district.get('district_total_votes'),
            'district_win_number': district.get('district_win_number'),
            'district_flip_number': district.get('district_flip_number')
        })
    
    return result


def format_for_llm_context_district_focused(extracted_data: List[Dict[str, Any]], candidate_context: Dict[str, Any]) -> str:
    """
    Format extracted election data for LLM consumption with district focus.
    Handles both district-based and statewide races.
    """
    context_parts = []
    context_parts.append("=" * 80)
    context_parts.append("ELECTION DATA ANALYSIS FOR YOUR CAMPAIGN")
    context_parts.append("=" * 80)
    context_parts.append("")
    context_parts.append(f"Candidate Information:")
    context_parts.append(f"  Office: {candidate_context.get('office_position', 'Unknown')}")
    context_parts.append(f"  District: {candidate_context.get('district_name', 'Unknown')}")
    context_parts.append(f"  Analysis Year: {candidate_context.get('current_year', 'Unknown')}")
    
    is_statewide = candidate_context.get('district_name') == 'Statewide'
    if is_statewide:
        context_parts.append(f"  Race Type: STATEWIDE (All Virginia precincts)")
    else:
        context_parts.append(f"  Race Type: DISTRICT-BASED")
    
    context_parts.append("")
    context_parts.append("=" * 80)
    context_parts.append("")
    
    # Separate candidate district data from contextual
    candidate_data = [d for d in extracted_data if d.get('_metadata', {}).get('category') == 'candidate_district']
    contextual_data = [d for d in extracted_data if d.get('_metadata', {}).get('category') == 'contextual']
    
    # Sort candidate data by year (most recent first)
    candidate_data.sort(key=lambda x: x.get('year', 0), reverse=True)
    
    # Format candidate district data
    if candidate_data:
        if is_statewide:
            context_parts.append("YOUR STATEWIDE ELECTION HISTORY (All Virginia Precincts):")
        else:
            context_parts.append("YOUR DISTRICT'S ELECTION HISTORY (Precinct-Level Detail):")
        context_parts.append("=" * 80)
        for data in candidate_data:
            context_parts.append(format_district_precinct_chunk(data, candidate_context))
        context_parts.append("")
    
    # Format contextual data
    if contextual_data:
        context_parts.append("CONTEXTUAL ELECTIONS (Top-of-Ticket Races):")
        context_parts.append("=" * 80)
        context_parts.append("These races provide context for turnout patterns and partisan trends.")
        context_parts.append("")
        for data in contextual_data:
            context_parts.append(format_contextual_chunk(data))
        context_parts.append("")
    
    return "\n".join(context_parts)


def format_district_precinct_chunk(data: Dict[str, Any], candidate_context: Dict[str, Any]) -> str:
    """
    Format district precinct-level data.
    Handles both district-based and statewide races.
    """
    metadata = data.get('_metadata', {})
    election_info = metadata.get('election_info', {})
    is_statewide = metadata.get('is_statewide', False)
    parts = []
    
    # Header
    year = data.get('year', 'N/A')
    stage = data.get('stage', 'N/A')
    is_pres_year = " (PRESIDENTIAL YEAR)" if election_info.get('is_presidential_year') else ""
    
    parts.append("\n" + "─" * 80)
    parts.append(f"📊 {data.get('office', 'Unknown')} {year} {stage}{is_pres_year}")
    parts.append("─" * 80)
    parts.append(f"Purpose: {metadata.get('purpose', 'N/A')}")
    
    # Check for errors
    if 'error' in metadata:
        parts.append(f"⚠️  ERROR: {metadata['error']}")
        return "\n".join(parts)
    
    if '_error' in data:
        parts.append(f"⚠️  {data['_error']}")
        return "\n".join(parts)
    
    district = data.get('district')
    if not district:
        parts.append("⚠️  No district data available")
        return "\n".join(parts)
    
    parts.append("")
    parts.append(f"District: {district.get('district_name', 'Unknown')}")
    parts.append(f"  📍 District Total Votes: {district.get('district_total_votes', 0):,}")
    
    # Only show win/flip numbers for district-based races
    if not is_statewide:
        if district.get('district_win_number'):
            parts.append(f"  🎯 Votes Needed to Win: {district.get('district_win_number'):,}")
        
        if district.get('district_flip_number'):
            parts.append(f"  🔄 Votes Needed to Flip: {district.get('district_flip_number'):,}")
    
    parts.append("")
    
    # Get precincts
    precincts = district.get('precincts', [])
    
    if is_statewide:
        parts.append(f"STATEWIDE PRECINCT DATA:")
        parts.append(f"Total Precincts: {len(precincts)}")
        parts.append("")
        
        # Group precincts by locality for statewide races
        locality_groups = {}
        for precinct in precincts:
            locality = precinct.get('locality', 'Unknown')
            if locality not in locality_groups:
                locality_groups[locality] = []
            locality_groups[locality].append(precinct)
        
        parts.append(f"Localities Covered: {len(locality_groups)}")
        parts.append("")
        
        # Show top 10 localities by total votes
        locality_totals = []
        for locality, precs in locality_groups.items():
            total = sum(p.get('precinct_total_votes', 0) for p in precs)
            locality_totals.append((locality, total, len(precs)))
        
        locality_totals.sort(key=lambda x: x[1], reverse=True)
        
        parts.append("Top 10 Localities by Turnout:")
        for i, (locality, total, count) in enumerate(locality_totals[:10], 1):
            parts.append(f"  {i}. {locality}: {total:,} votes ({count} precincts)")
        
        parts.append("")
        parts.append(f"Note: Full precinct data available for all {len(precincts)} precincts statewide.")
        parts.append("For detailed precinct analysis, filter by specific localities in your query.")
        
    else:
        # District-based race - show all precincts
        parts.append("PRECINCT BREAKDOWN:")
        parts.append("")
        
        # Sort precincts by total votes (highest first)
        precincts_sorted = sorted(precincts, key=lambda x: x.get('precinct_total_votes', 0), reverse=True)
        
        for i, precinct in enumerate(precincts_sorted, 1):
            precinct_name = precinct.get('precinct_name', 'Unknown')
            locality = precinct.get('locality', 'Unknown')
            precinct_total = precinct.get('precinct_total_votes', 0)
            
            parts.append(f"  {i}. {precinct_name} ({locality})")
            parts.append(f"     Total Votes: {precinct_total:,}")
            
            if precinct.get('win_number'):
                parts.append(f"     Win Number: {precinct.get('win_number'):,}")
            
            if precinct.get('flip_number'):
                parts.append(f"     Flip Number: {precinct.get('flip_number'):,}")
            
            # Candidate results
            results = precinct.get('results', [])
            if results:
                results_sorted = sorted(results, key=lambda x: x.get('votes', 0), reverse=True)
                parts.append(f"     Candidate Results:")
                
                for j, result in enumerate(results_sorted, 1):
                    candidate = result.get('candidate_name', 'Unknown')
                    party = result.get('party', '')
                    votes = result.get('votes', 0)
                    percentage = (votes / precinct_total * 100) if precinct_total > 0 else 0
                    
                    indicator = "🏆" if j == 1 else "  "
                    party_str = f" ({party})" if party else ""
                    parts.append(f"       {indicator} {j}. {candidate}{party_str}: {votes:,} votes ({percentage:.1f}%)")
                
                # Calculate margin
                if len(results_sorted) >= 2:
                    margin = results_sorted[0].get('votes', 0) - results_sorted[1].get('votes', 0)
                    margin_pct = (margin / precinct_total * 100) if precinct_total > 0 else 0
                    parts.append(f"     Margin: {margin:,} votes ({margin_pct:.1f}%)")
            
            parts.append("")
        
        # District-level summary
        total_district_votes = district.get('district_total_votes', 0)
        if total_district_votes > 0:
            parts.append("DISTRICT SUMMARY:")
            parts.append(f"  • Total precincts analyzed: {len(precincts)}")
            parts.append(f"  • Total district votes: {total_district_votes:,}")
            parts.append(f"  • Average votes per precinct: {total_district_votes // len(precincts) if precincts else 0:,}")
    
    parts.append("")
    
    return "\n".join(parts)


def format_contextual_chunk(data: Dict[str, Any]) -> str:
    """Format contextual election data."""
    metadata = data.get('_metadata', {})
    election_info = metadata.get('election_info', {})
    parts = []
    
    # Header
    office = election_info.get('office', data.get('office', 'Unknown'))
    year = data.get('year', 'N/A')
    stage = data.get('stage', 'N/A')
    
    parts.append(f"\n{office} {year} {stage}")
    parts.append(f"Purpose: {metadata.get('purpose', 'N/A')}")
    
    if 'error' in metadata:
        parts.append(f"ERROR: {metadata['error']}")
        return "\n".join(parts)
    
    # Format based on extraction level
    level = metadata.get('extraction_level', '')
    
    if level == 'FILE_LEVEL':
        parts.append(f"  Total Votes: {data.get('total_votes', 'N/A'):,}")
    
    elif level == 'DISTRICT_SUMMARY':
        parts.append(f"  Total Votes: {data.get('total_votes', 'N/A'):,}")
        parts.append(f"  Number of Districts: {len(data.get('districts', []))}")
        
        districts = data.get('districts', [])
        if districts:
            parts.append("  Top 5 Districts by Turnout:")
            sorted_districts = sorted(districts, key=lambda x: x.get('district_total_votes', 0), reverse=True)[:5]
            for dist in sorted_districts:
                parts.append(f"    • {dist.get('district_name', 'Unknown')}: {dist.get('district_total_votes', 0):,} votes")
    
    parts.append("")
    return "\n".join(parts)


def create_llm_prompt_with_context_district_focused(user_query: str, extracted_data: List[Dict[str, Any]], candidate_context: Dict[str, Any]) -> str:
    """
    Create the final prompt for the LLM with district-focused context.
    """
    context = format_for_llm_context_district_focused(extracted_data, candidate_context)
    
    is_statewide = candidate_context.get('district_name') == 'Statewide'
    race_type = "statewide" if is_statewide else "district"
    
    prompt = f"""You are an expert political campaign strategist providing insights to a candidate running for office.

CANDIDATE PROFILE:
- Office: {candidate_context.get('office_position')}
- District: {candidate_context.get('district_name')}
- Election Year: {candidate_context.get('current_year')}
- Race Type: {race_type.upper()}

CANDIDATE'S QUESTION:
{user_query}

ELECTION DATA ANALYSIS:
{context}

INSTRUCTIONS:
Based on the precinct-level election data provided above, provide a comprehensive, strategic answer to their question. Your answer should:

1. Be specific and actionable - use actual numbers from the data
2. Identify patterns and trends across the historical period
3. {"For statewide races: Identify key regions/localities and their voting patterns" if is_statewide else "Highlight which precincts are base strongholds, swing precincts, or opponent strongholds"}
4. Note presidential year vs off-year turnout differences
5. Provide strategic recommendations based on the data
6. Reference specific {"localities and regional patterns" if is_statewide else "precincts, vote totals, and margins"} when relevant
7. Consider how top-of-ticket races (Presidential, Governor) affected performance

Be direct, strategic, and data-driven. This candidate needs actionable insights to win their race.

STRATEGIC ANALYSIS:"""
    
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
    """
    # Step 1: Get retrieval plan from Bedrock
    logger.info("Step 1: Getting retrieval plan from Bedrock...")
    retrieval_plan = get_retrieval_plan_from_bedrock(user_query, candidate_context=candidate_context, model_id=model_id)
    logger.debug(json.dumps(retrieval_plan, indent=2))
    
    # Step 2: Extract data from S3
    logger.info("Step 2: Extracting data from S3...")
    extracted_data = extract_data_from_s3_district_focused(
        retrieval_plan,
        candidate_context['district_name'],
        candidate_context['office_position']
    )
    
    # Step 3: Format for LLM
    logger.info("Step 3: Formatting data for LLM...")
    final_prompt = create_llm_prompt_with_context_district_focused(
        user_query,
        extracted_data,
        candidate_context
    )

    with open("final_prompt.txt", 'w') as f:
        f.write(final_prompt)
    
    # Step 4: Get final answer
    logger.info("Step 4: Getting final answer from Bedrock...")
    answer = get_answer_from_bedrock(final_prompt, model_id=model_id)
    # answer = 'testing'
    
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
    # Example 1: House of Delegates candidate (district-based)
    candidate_context_hod = {
        "office_position": "House_of_Delegates",
        "district_name": "District_41",
        "current_year": 2025
    }
    user_query_hod = 'How can I, a democratic candidate, win the election?'
    
    # Example 2: Governor candidate (statewide)
    candidate_context_gov = {
        "office_position": "Governor",
        "district_name": "Statewide",
        "current_year": 2025
    }
    user_query_gov = 'What are the key factors for winning the governor race in Virginia?'
    
    # Run the appropriate example
    final_response = main(user_query=user_query_hod, candidate_context=candidate_context_hod, model_id=MODEL_ID)
    
    logger.info(f"\n\n{final_response['body']['answer']}")
    logger.info(f"\n\nData sources used: {final_response['body']['data_sources']}")