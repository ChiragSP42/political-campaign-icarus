"""
Lambda Function: generate-insights

Purpose:
  Retrieves completed questionnaire from DynamoDB Streams event and generates insights.
  Lambda is triggered by DynamoDB Streams on the Questionnaire table.
  Generated insights are stored in the Main DynamoDB table with SK = INSIGHTS.
  
Input: DynamoDB Streams event (NEW_IMAGE) from Questionnaire table
  
Output:
{
    'statusCode': 200,
    'body': json.dumps({
        'success': True,
        'message': 'Generated insights for {userId}',
        'userId': <userId>,
    }),
    'headers': {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
    }
}
"""
from helpers import *
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
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from boto3.dynamodb.types import TypeDeserializer

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
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')
dynamodb = boto3.resource('dynamodb')
deserializer = TypeDeserializer()

# Environment variables
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


def deserialize_dynamodb_record(record: Dict) -> Dict:
    """Convert DynamoDB stream record format to plain Python dict."""
    return {k: deserializer.deserialize(v) for k, v in record.items()}


def lambda_handler(event, context):
    """
    Main Lambda handler function triggered by DynamoDB Streams on the Questionnaire table.
    Parses the new questionnaire record, generates insights, and saves them to the Main table.
    """
    
    try:
        for record in event.get('Records', []):
            # Only process INSERT and MODIFY events
            event_name = record.get('eventName')
            if event_name not in ('INSERT', 'MODIFY'):
                print(f"Skipping event: {event_name}")
                continue

            # Get the new image from the stream record
            new_image = record.get('dynamodb', {}).get('NewImage')
            if not new_image:
                print("No NewImage in stream record, skipping")
                continue

            # Deserialize DynamoDB types to plain Python types
            item = deserialize_dynamodb_record(new_image)
            print(f"Received DynamoDB stream record: {json.dumps(item, default=str)}")

            user_id = item.get('userId')
            if not user_id:
                print("No userId found in record, skipping")
                continue

            # Extract questionnaire answers (everything except ignored fields)
            questionnaire_answers = {k: v for k, v in item.items() if k not in IGNORED_FIELDS}
            print(f"Questionnaire answers for {user_id}: {json.dumps(questionnaire_answers, default=str)}")

            # Generate insights
            generated_insights = call_chatbot_logic(
                bedrock_agent_runtime=bedrock_agent_runtime,
                s3_client=s3_client,
                bedrock_runtime=bedrock_runtime,
                questionnaire=questionnaire_answers,
                model_id=MODEL_ID,
                scope_model_id=MODEL_ID
            )

            if not generated_insights:
                print(f"No insights generated for {user_id}")
                continue

            # Save generated insights to Main DynamoDB table
            try:
                main_table.put_item(Item={
                    'userId': user_id,
                    'SK': 'INSIGHTS',
                    'insights': generated_insights,
                    'generatedAt': datetime.now().isoformat(),
                })
                print(f"Saved insights for {user_id} to Main table")
            except Exception as e:
                print(f"Failed to save insights to Main table: {str(e)}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'message': 'Insights generation complete',
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
    
    except json.JSONDecodeError:
        return error_response(400, 'Invalid JSON in request body')
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return error_response(500, str(e))

def call_chatbot_logic(bedrock_runtime: Any, 
                       bedrock_agent_runtime: Any, 
                       s3_client: Any,
                       questionnaire: Dict,
                       model_id: str,
                       scope_model_id: Optional[str]) -> str:
    
    user_context_string = ""
    try:
        # Step 2: Prepare context from questionnaire
        user_context_string = prepare_user_context(questionnaire)
    except Exception as e:
        print('Problem preparing context from questionnaire: {str(e)}')
        return ""
    
    scope_model_id = scope_model_id if scope_model_id else model_id
    chatbot = PCMChatbot(scope_model_id=model_id,
                        model_id=model_id,
                        s3_client=s3_client,
                        bedrock_runtime=bedrock_runtime,
                        bedrock_agent_runtime=bedrock_agent_runtime,
                        candidate_context=questionnaire)
    
    election_data = []
    try:
        # Load candidates election data and other election's data for use.
        candidate_election_data, other_election_data = chatbot.load_data()
        if candidate_election_data and other_election_data:
            election_data = candidate_election_data + other_election_data
            print("Loaded election data")
    except Exception as e:
        print(f'Error loading election data: {str(e)}')
        return ""
    
    textual_data = ""
    try:
        # Formatting data for LLM...
        textual_data = chatbot.create_llm_prompt_with_context_all_elections(election_data=election_data,
                                                                            candidate_context=user_context_string)
        if textual_data:
            print(f"Textual data: {textual_data[:20]}")
    except Exception as e:
        print(f"Error formatting data for LLM: {str(e)}")
        return ""

    # (N+1) Strategy to combat token ceiling
    while True:
        if count_tokens(prompt=textual_data):
            print("All data fits inside one Bedrock call")
            try:
                # Final response from Bedrock...
                print("Final response from Bedrock...")
                response = chatbot.get_answer_from_bedrock(prompt=textual_data, progress=None)
                answer = response['output']['message']['content'][0]['text']
                return answer
            except Exception as e:
                print(f"Error getting final response from Bedrock: {str(e)}")
                return "Too data to handle, could not generate insights"
        else:
            print("(N+1) strategy needs to be implemented")
            result = split_counter(query=textual_data)
            enriched_texts = []
            start_index = 0
            context = chatbot._format_for_llm_context_all_elections(election_data)
            for i in range(result['split']):
                end_index = start_index + result['parts'] + (1 if i < result['remainder'] else 0)
                extracted_data_subset = context[start_index: end_index]
                textual_data = chatbot.create_llm_prompt_with_context_all_elections(election_data=election_data,
                                                                            candidate_context=user_context_string)
                enriched_texts.append(extracted_data_subset)
                start_index = end_index
            print("Split counting done, now starting concurrent LLM calls")
            max_workers = result['split'] if result['split'] <= 3 else 3
            progress = ProgressTracker(result['split'])
            start_time = time.time()
            filled_parts = []
            print("Running concurrent calls")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_completed = [executor.submit(chatbot.get_answer_from_bedrock, 
                                                    enriched_text,
                                                    progress) for enriched_text in enriched_texts]

                for future in as_completed(future_completed):
                    result = future.result()
                    filled_parts.append(result)

            elapsed_time = time.time() - start_time
            print(f"Time elapsed for concurrent LLM calls: {elapsed_time:.2f} seconds")
                # print(f"Average time per split: {(elapsed_time/result['split']):.2f} seconds/split")

            if filled_parts:
                # Stitch the part answers and pass it through one final LLM to polish everything out
                stitched = "\n".join(filled_parts)
                print("Final LLM call to polish it out")
                tokens = count_tokens(f"{stitched}\n\nThe above content contains insights generated in chunks due to it's size, I want you to rewrite it as one whole piece")
                print(f"Total input tokens after everything: {tokens}")
                start_time = time.time()
                try:
                    final_output = bedrock_runtime.converse(modelId=MODEL_ID,
                                                                                messages=[
                                                                                    {
                                                                                        'role': 'user',
                                                                                        'content': [
                                                                                            {
                                                                                                'text': f"{stitched}\n\nThe above content contains insights generated in chunks due to it's size, I want you to rewrite it as one whole piece"
                                                                                            }
                                                                                        ]
                                                                                    }
                                                                                ])
                    elapsed_time = time.time() - start_time
                    print(f"Time elapsed for polishing LLM: {elapsed_time:.2f} seconds")
                    return final_output['output']['message']['content'][0]['text']
                except Exception as e:
                    print("Too much data to handle in final polishing")
                    return "Too much data to process, please try with a different office/district"
            else:
                return "N+1 failed, look into it"
    

def prepare_user_context(questionnaire: dict) -> str:
    """
    Convert questionnaire data into context for the LLM.
    
    Extract:
    - Office running for
    - District
    - Background info (military, public service, etc.)
    - Communication archetype (Firebrand, Bridge-Builder, etc.)
    """
    
    answers = questionnaire.get('answers', {})
    
    context = []
    for key, value in questionnaire.items():
        if key == "fullName":
            format = f'Full name of candidate: {value}'
            context.append(format)
        elif key == 'district_name':
            format = f'District candidate is running for: {value}'
            context.append(format)
        elif key == 'office_position':
            format = f'Office candidate is running for: {value}'
            context.append(format)
        else:
            format = f"Question: {key}\nAnswer: {value}"
            context.append(format)

    context = "\n".join(context)
    return context