"""
Lambda Function: chatbot

Purpose:
  Interactive chatbot backend
  
Input Body (JSON):
{
    "query": The user's query,
    "coversational_history": The user's present conversational history,
    "email": The user's email ID for logically seperated stuff
}
  
Output:
{
    'statusCode': 200,
    'body': json.dumps({
        'success': True,
        'message': 'Generated insights for {email}',
        'email': <email>,
    }),
    'headers': {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
    }
}
"""

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
from botocore.exceptions import ClientError

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

# Environment variables
ACCOUNT_ID = sts_client.get_caller_identity()['Account']
S3_GENERATED_INSIGHTS = os.getenv("S3_GENERATED_INSIGHTS", 'generated-insights')
S3_GENERATED_INSIGHTS = f"{S3_GENERATED_INSIGHTS}-{ACCOUNT_ID}"
S3_RESPONSES = os.getenv("S3_RESPONSES", 'chatbot-responses')
S3_RESPONSES = f"{S3_RESPONSES}-{ACCOUNT_ID}"
S3_QUESTIONNAIRES = os.getenv("S3_QUESTIONNAIRES", 'icarus-questionnaires')
S3_QUESTIONNAIRES = f"{S3_QUESTIONNAIRES}-{ACCOUNT_ID}"
CHATBOT_PROMPT = os.getenv("CHATBOT_PROMPT", 'campaign_advisor_prompt.md')
PROMPT_BUCKET = os.getenv("PROMPT_BUCKET", 'prompt-bucket')
PROMPT_BUCKET = f"{PROMPT_BUCKET}-{ACCOUNT_ID}"
MODEL_ID = os.environ.get('MODEL_ID', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')
KB_ID = os.environ.get('KB_ID', '')


def lambda_handler(event, context):
    """
    Main Lambda handler function for saving questionnaire.

    The event format: {'email': The email ID, 'answers': The questionnaire answers}
    """
    print("Event", event)
    print(type(event))
    try:
        # Parse the request body
        body = event.get('body', {})
        print(f"Body: {body}")
        user_query = body.get("query", "")
        conversation_history = body.get("conversation_history", [])
        email = body.get("email", "")
        username = email.split("@")[0]

        # Get system prompt for chatbot
        print("Getting system prompt for chatbot")
        response = s3_client.get_object(Bucket=PROMPT_BUCKET,
                                        Key=CHATBOT_PROMPT)
        chatbot_prompt = response['Body'].read().decode('utf-8')

        # Get relevant election law chunks from KB
        # election_laws = retrieve_laws(user_query=user_query)

        # Get insights for candidate
        print("Getting insights for candidate")
        response = s3_client.get_object(Bucket=S3_GENERATED_INSIGHTS,
                                        Key=f'{username}/{username}_insights.md')
        user_insights = response['Body'].read().decode('utf-8')

        # Get questionnaire for candidate
        print("Getting questionnaire for candidate")
        response = s3_client.get_object(Bucket=S3_QUESTIONNAIRES,
                                        Key=f"{username}/{username}_questionnaire.json")
        questionnaire = json.loads(response['Body'].read())

        questionnaire_text = prepare_user_context(questionnaire=questionnaire)
        s3_client.put_object(Bucket=S3_GENERATED_INSIGHTS, Key=f'questionnaire_text.md', Body=chatbot_prompt, ContentType='text/markdown')

        # Fill system prompt
        print("Filling chatbot prompt")
        chatbot_prompt = chatbot_prompt.replace("{candidate_questionnaire}", questionnaire_text)
        chatbot_prompt = chatbot_prompt.replace("{generated_insights}", user_insights)
        chatbot_prompt = chatbot_prompt.replace("{user_query}", user_query)
        # chatbot_prompt = chatbot_prompt.replace("{relevant_election_laws}", election_laws)
        # Format query into message format.
        s3_client.put_object(Bucket=S3_GENERATED_INSIGHTS, Key=f'chatbot_prompt.md', Body=chatbot_prompt, ContentType='text/markdown')
        message = {
            'role': 'user',
            'content': [{'text': chatbot_prompt}]
        }

        messages = conversation_history + [message]

        print("Converse call")
        response = bedrock_runtime.converse(
            modelId=MODEL_ID,
            messages=messages,
            inferenceConfig={
                'temperature': 0.3
            }
        )

        answer = response['output']['message']['content'][0]['text']
        s3_client.put_object(Bucket=S3_RESPONSES, Key=f'{username}/{username}_response.md', Body=answer, ContentType='text/markdown')
        print("LLM generation successful")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'message': answer,
                'email': email
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

def retrieve_laws(user_query: str) -> str:
    response = bedrock_agent_runtime.retrieve(
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
    for key, value in answers.items():
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

    context = "\n\n".join(context)
    
    return context