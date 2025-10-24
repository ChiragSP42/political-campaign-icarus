import json
import boto3
import os
from aws_helpers import helpers
from aws_helpers import helpers
from typing import Dict, List, Any
from dotenv import load_dotenv
load_dotenv(override=True)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY,
                        region_name='us-east-1')
bedrock_runtime = session.client('bedrock-runtime')
s3_client = session.client('s3')
logger = helpers._setup_logger('idk', level=10)

def extract_data_from_s3(retrieval_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract data from S3 based on the retrieval plan from Bedrock.
    
    Args:
        retrieval_plan: List of retrieval specifications from Bedrock model
        
    Returns:
        List of extracted data chunks
    """
    extracted_data = []
    
    for plan_item in retrieval_plan:
        s3_path = plan_item['s3_path']
        extraction_spec = plan_item['extraction_spec']
        category = plan_item['category']
        purpose = plan_item['purpose']
        logger.info(f"\x1b[33mExtracting data from: {s3_path}\x1b[0m")
        
        # Parse S3 path (format: bucket_name is assumed to be configured)
        # If your path includes bucket: bucket/path/to/file.json, adjust accordingly
        bucket_name = "predictif-election-data"  # Configure this
        
        try:
            # Fetch the full JSON from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_path)
            full_data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Extract based on level
            extracted_chunk = extract_by_level(
                full_data, 
                extraction_spec['level'],
                extraction_spec.get('fields', []),
                extraction_spec.get('filters', {})
            )
            
            # Add metadata
            extracted_chunk['_metadata'] = {
                'source_path': s3_path,
                'category': category,
                'purpose': purpose,
                'extraction_level': extraction_spec['level']
            }
            
            extracted_data.append(extracted_chunk)
            
        except Exception as e:
            logger.error(f"\x1b[31mError extracting data from {s3_path}: {str(e)}\x1b[0m")
    
    return extracted_data


def extract_by_level(data: Dict, level: str, fields: List[str], filters: Dict) -> Dict:
    """
    Extract data based on the specified extraction level.
    
    Args:
        data: Full JSON data from S3
        level: Extraction level (FILE_LEVEL, DISTRICT_SUMMARY, etc.)
        fields: List of fields to extract
        filters: Dictionary of filter criteria
        
    Returns:
        Extracted data chunk
    """
    if level == "FILE_LEVEL":
        return extract_file_level(data)
    
    elif level == "DISTRICT_SUMMARY":
        return extract_district_summary(data)
    
    elif level == "DISTRICT_FILTERED":
        return extract_district_filtered(data, filters.get('district_names', []))
    
    elif level == "PRECINCT_FILTERED":
        return extract_precinct_filtered(data, filters.get('precinct_names', []))
    
    elif level == "CANDIDATE_RESULTS":
        return extract_candidate_results(data)
    
    elif level == "CANDIDATE_FILTERED":
        return extract_candidate_filtered(data, filters.get('candidate_names', []))
    
    else:
        raise ValueError(f"Unknown extraction level: {level}")


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


def extract_district_filtered(data: Dict, district_names: List[str]) -> Dict:
    """Extract specific districts with full precinct details."""
    result = {
        'record_id': data.get('record_id'),
        'office': data.get('office'),
        'year': data.get('year'),
        'stage': data.get('stage'),
        'total_votes': data.get('total_votes'),
        'districts': []
    }
    
    for district in data.get('districts', []):
        if district.get('district_name') in district_names:
            result['districts'].append(district)
    
    return result


def extract_precinct_filtered(data: Dict, precinct_names: List[str]) -> Dict:
    """Extract specific precincts across all districts."""
    result = {
        'record_id': data.get('record_id'),
        'office': data.get('office'),
        'year': data.get('year'),
        'stage': data.get('stage'),
        'total_votes': data.get('total_votes'),
        'districts': []
    }
    
    for district in data.get('districts', []):
        filtered_precincts = [
            p for p in district.get('precincts', [])
            if p.get('precinct_name') in precinct_names
        ]
        
        if filtered_precincts:
            result['districts'].append({
                'district_name': district.get('district_name'),
                'district_total_votes': district.get('district_total_votes'),
                'precincts': filtered_precincts
            })
    
    return result


def extract_candidate_results(data: Dict) -> Dict:
    """Extract only candidate names and vote counts."""
    result = {
        'record_id': data.get('record_id'),
        'office': data.get('office'),
        'year': data.get('year'),
        'stage': data.get('stage'),
        'districts': []
    }
    
    for district in data.get('districts', []):
        district_data = {
            'district_name': district.get('district_name'),
            'precincts': []
        }
        
        for precinct in district.get('precincts', []):
            precinct_data = {
                'precinct_name': precinct.get('precinct_name'),
                'results': precinct.get('results', [])
            }
            district_data['precincts'].append(precinct_data)
        
        result['districts'].append(district_data)
    
    return result


def extract_candidate_filtered(data: Dict, candidate_names: List[str]) -> Dict:
    """Extract results only for specific candidates."""
    result = {
        'record_id': data.get('record_id'),
        'office': data.get('office'),
        'year': data.get('year'),
        'stage': data.get('stage'),
        'districts': []
    }
    
    for district in data.get('districts', []):
        district_data = {
            'district_name': district.get('district_name'),
            'precincts': []
        }
        
        for precinct in district.get('precincts', []):
            filtered_results = [
                r for r in precinct.get('results', [])
                if r.get('candidate_name') in candidate_names
            ]
            
            if filtered_results:
                precinct_data = {
                    'precinct_name': precinct.get('precinct_name'),
                    'precinct_total_votes': precinct.get('precinct_total_votes'),
                    'results': filtered_results
                }
                district_data['precincts'].append(precinct_data)
        
        if district_data['precincts']:
            result['districts'].append(district_data)
    
    return result

def format_for_llm_context(extracted_data: List[Dict[str, Any]]) -> str:
    """
    Format extracted election data into a readable text format for LLM consumption.
    
    Args:
        extracted_data: List of extracted data chunks from extract_data_from_s3()
        
    Returns:
        Formatted string ready to be passed as context to an LLM
    """
    context_parts = []
    context_parts.append("=" * 80)
    context_parts.append("ELECTION DATA CONTEXT")
    context_parts.append("=" * 80)
    context_parts.append("")
    
    # Group by category
    primary_data = [d for d in extracted_data if d.get('_metadata', {}).get('category') == 'primary']
    contextual_data = [d for d in extracted_data if d.get('_metadata', {}).get('category') == 'contextual']
    historical_data = [d for d in extracted_data if d.get('_metadata', {}).get('category') == 'historical']
    
    # Format primary data
    if primary_data:
        context_parts.append("PRIMARY DATA (Directly answers the query):")
        context_parts.append("-" * 80)
        for data in primary_data:
            context_parts.append(format_single_chunk(data))
        context_parts.append("")
    
    # Format contextual data
    if contextual_data:
        context_parts.append("CONTEXTUAL DATA (Same-year elections for context):")
        context_parts.append("-" * 80)
        for data in contextual_data:
            context_parts.append(format_single_chunk(data))
        context_parts.append("")
    
    # Format historical data
    if historical_data:
        context_parts.append("HISTORICAL DATA (Past years for trend analysis):")
        context_parts.append("-" * 80)
        for data in historical_data:
            context_parts.append(format_single_chunk(data))
        context_parts.append("")
    
    return "\n".join(context_parts)


def format_single_chunk(data: Dict[str, Any]) -> str:
    """Format a single data chunk based on its extraction level."""
    metadata = data.get('_metadata', {})
    parts = []
    
    # Header
    parts.append(f"\n[{metadata.get('category', 'unknown').upper()}] {data.get('office', 'Unknown')} "
                f"{data.get('year', 'N/A')} {data.get('stage', 'N/A')}")
    parts.append(f"Source: {metadata.get('source_path', 'Unknown')}")
    parts.append(f"Purpose: {metadata.get('purpose', 'N/A')}")
    parts.append(f"Extraction Level: {metadata.get('extraction_level', 'N/A')}")
    
    # Check for errors
    if 'error' in metadata:
        parts.append(f"ERROR: {metadata['error']}")
        return "\n".join(parts)
    
    parts.append("")
    
    # Format based on extraction level
    extraction_level = metadata.get('extraction_level', '')
    
    if extraction_level == "FILE_LEVEL":
        parts.append(f"Total Votes: {data.get('total_votes', 'N/A'):,}")
    
    elif extraction_level == "DISTRICT_SUMMARY":
        parts.append(f"Total Votes: {data.get('total_votes', 'N/A'):,}")
        parts.append(f"Number of Districts: {len(data.get('districts', []))}")
        parts.append("")
        parts.append("District-Level Summary:")
        for district in data.get('districts', []):
            parts.append(f"  • {district.get('district_name', 'Unknown')}")
            parts.append(f"    - Total Votes: {district.get('district_total_votes', 'N/A'):,}")
            if district.get('district_win_number'):
                parts.append(f"    - Win Number: {district.get('district_win_number'):,}")
            if district.get('district_flip_number'):
                parts.append(f"    - Flip Number: {district.get('district_flip_number'):,}")
    
    elif extraction_level in ["DISTRICT_FILTERED", "PRECINCT_FILTERED"]:
        parts.append(f"Total Votes: {data.get('total_votes', 'N/A'):,}")
        parts.append("")
        for district in data.get('districts', []):
            parts.append(f"District: {district.get('district_name', 'Unknown')}")
            parts.append(f"  District Total Votes: {district.get('district_total_votes', 'N/A'):,}")
            if district.get('district_win_number'):
                parts.append(f"  District Win Number: {district.get('district_win_number'):,}")
            if district.get('district_flip_number'):
                parts.append(f"  District Flip Number: {district.get('district_flip_number'):,}")
            parts.append("")
            
            for precinct in district.get('precincts', []):
                parts.append(f"  Precinct: {precinct.get('precinct_name', 'Unknown')}")
                parts.append(f"    Total Votes: {precinct.get('precinct_total_votes', 'N/A'):,}")
                if precinct.get('win_number'):
                    parts.append(f"    Win Number: {precinct.get('win_number'):,}")
                if precinct.get('flip_number'):
                    parts.append(f"    Flip Number: {precinct.get('flip_number'):,}")
                
                if precinct.get('results'):
                    parts.append(f"    Candidate Results:")
                    for result in precinct.get('results', []):
                        candidate = result.get('candidate_name', 'Unknown')
                        votes = result.get('votes', 0)
                        percentage = (votes / precinct.get('precinct_total_votes', 1)) * 100
                        parts.append(f"      - {candidate}: {votes:,} votes ({percentage:.1f}%)")
                parts.append("")
    
    elif extraction_level in ["CANDIDATE_RESULTS", "CANDIDATE_FILTERED"]:
        parts.append("Candidate Vote Results:")
        parts.append("")
        
        # Aggregate candidate totals across all districts/precincts
        candidate_totals = {}
        
        for district in data.get('districts', []):
            for precinct in district.get('precincts', []):
                for result in precinct.get('results', []):
                    candidate = result.get('candidate_name', 'Unknown')
                    votes = result.get('votes', 0)
                    candidate_totals[candidate] = candidate_totals.get(candidate, 0) + votes
        
        # Sort by votes descending
        sorted_candidates = sorted(candidate_totals.items(), key=lambda x: x[1], reverse=True)
        
        for candidate, total_votes in sorted_candidates:
            parts.append(f"  • {candidate}: {total_votes:,} votes")
        
        # Also show district/precinct breakdown
        parts.append("")
        parts.append("District/Precinct Breakdown:")
        for district in data.get('districts', []):
            parts.append(f"  {district.get('district_name', 'Unknown')}:")
            for precinct in district.get('precincts', []):
                parts.append(f"    {precinct.get('precinct_name', 'Unknown')}:")
                for result in precinct.get('results', []):
                    parts.append(f"      - {result.get('candidate_name', 'Unknown')}: "
                               f"{result.get('votes', 0):,} votes")
    
    parts.append("")
    parts.append("-" * 40)
    
    return "\n".join(parts)


def create_llm_prompt_with_context(user_query: str, extracted_data: List[Dict[str, Any]]) -> str:
    """
    Create the final prompt to send to the LLM with all extracted data as context.
    
    Args:
        user_query: The original user query
        extracted_data: List of extracted data chunks
        
    Returns:
        Complete prompt ready for LLM
    """
    context = format_for_llm_context(extracted_data)
    
    prompt = f"""You are a political campaign data analyst. Using the election data provided below, answer the user's question comprehensively and accurately.

USER QUESTION:
{user_query}

{context}

Based on the election data provided above, please provide a detailed analysis that answers the user's question. Include specific numbers, trends, and insights where relevant. If you notice any patterns related to presidential year effects, district-level shifts, or turnout variations, highlight those in your analysis.

ANSWER:"""
    
    return prompt

def get_retrieval_plan_from_bedrock(user_query: str) -> dict:
    """
    Call Bedrock to get the retrieval plan.
    """
    # Load the system prompt from the text file above
    with open('content_retrieval_S3_data_chunk_prompt.txt', 'r') as f:
        system_prompt = f.read()
    
    # Replace placeholder with actual query
    full_prompt = system_prompt.replace('{{USER_QUERY}}', user_query)
    
    # Call Bedrock Converse API
    response = bedrock_runtime.converse(
        modelId='us.anthropic.claude-3-5-sonnet-20241022-v2:0',  # or your preferred model
        messages=[
            {
                'role': 'user',
                'content': [{'text': full_prompt}]
            }
        ],
        inferenceConfig={
            'temperature': 0.1,
            'maxTokens': 4000
        }
    )
    
    # Extract the JSON response
    response_text = response['output']['message']['content'][0]['text']
    
    # Parse JSON from response
    # The model should return pure JSON, but handle cases where it adds explanation
    try:
        retrieval_plan = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'``````', response_text, re.DOTALL)
        if json_match:
            retrieval_plan = json.loads(json_match.group(1))
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                retrieval_plan = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not parse JSON from Bedrock response")
    
    return retrieval_plan


def get_answer_from_bedrock(prompt: str) -> str:
    """
    Call Bedrock to get the final answer based on extracted data.
    """
    response = bedrock_runtime.converse(
        modelId='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        messages=[
            {
                'role': 'user',
                'content': [{'text': prompt}]
            }
        ],
        inferenceConfig={
            'temperature': 0.3,
            'maxTokens': 4000
        }
    )
    
    return response['output']['message']['content'][0]['text']


def main(user_query):
        
    # Step 1: Call Bedrock to get retrieval plan
    logger.info("Step 1: Getting retrieval plan from Bedrock...")
    retrieval_plan = get_retrieval_plan_from_bedrock(user_query)
    
    # Step 2: Extract data from S3 based on retrieval plan
    logger.info("Step 2: Extracting data from S3...")
    extracted_data = extract_data_from_s3(retrieval_plan['retrieval_plan'])
    logger.debug(json.dumps(extracted_data, indent=2))
    
    # Step 3: Format data for LLM context
    logger.info("Step 3: Formatting data for LLM...")
    final_prompt = create_llm_prompt_with_context(user_query, extracted_data)
    
    # Step 4: Call Bedrock again to answer the question
    logger.info("Step 4: Getting final answer from Bedrock...")
    answer = get_answer_from_bedrock(final_prompt)
    
    return {
        'statusCode': 200,
        'body': {
            'answer': answer,
            'retrieval_plan': retrieval_plan,
            'data_sources': [d['_metadata']['source_path'] for d in extracted_data if '_metadata' in d]
        }
    }


if __name__ == "__main__":
    user_query = 'Office position: House of Delegates\nYear: 2023\n\nHow can I, a democratic candidate, win the elections for the House of Delegates for the district 41 [in the state of virginia]?'
    # user_query = '\n\nCan you help me analyze the numbers for the last year election for precinct Chincoteague? I want to win this election as democratic candidate?'
    final_response = main(user_query=user_query)
    logger.info(f"\n\n{final_response['body']['answer']}")
    # logger.info(f"\n\nData sources used: {final_response['body']['data_sources']}")