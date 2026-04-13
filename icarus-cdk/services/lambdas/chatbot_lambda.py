"""
Lambda Function: chatbot

Purpose:
  Interactive chatbot backend. Loads conversation history from DynamoDB
  and persists assistant responses.
  
Input Body (JSON):
{
    "query": The user's query,
    "chatId": The chat session ID (UUID),
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
from datetime import datetime, date, timezone
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
dynamodb = boto3.resource("dynamodb")

# Environment variables
ACCOUNT_ID = sts_client.get_caller_identity()['Account']

MAIN_TABLE_NAME = os.getenv("MAIN_TABLE_NAME")
QUESTIONNAIRE_TABLE_NAME = os.getenv("QUESTIONNAIRE_TABLE_NAME")
CHATBOT_PROMPT = os.getenv("CHATBOT_PROMPT", 'campaign_advisor_prompt.md')
PROMPT_BUCKET = os.getenv("PROMPT_BUCKET", 'prompt-bucket')
PROMPT_BUCKET = f"{PROMPT_BUCKET}-{ACCOUNT_ID}"
MODEL_ID = os.environ.get('MODEL_ID', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')
KB_ID = os.environ.get('KB_ID', '')
CHAT_HISTORY_TABLE = os.getenv("CHAT_HISTORY_TABLE", "")

# DynamoDB table reference
chat_history_table = dynamodb.Table(CHAT_HISTORY_TABLE) if CHAT_HISTORY_TABLE else None
main_table = dynamodb.Table(MAIN_TABLE_NAME)
questionnaire_table = dynamodb.Table(QUESTIONNAIRE_TABLE_NAME)

IGNORED_FIELDS = {'userId', 'savedAt', 'updatedAt'}


def lambda_handler(event, context):
    """
    Main Lambda handler function for saving questionnaire.

    The event format: {'email': The email ID, 'answers': The questionnaire answers}
    """
    print("Event", event)
    try:
        # Parse the request body
        body = event.get('body', {})
        print(f"Body: {body}")
        user_query = body.get("query", "")
        chat_id = body.get("chatId", "")
        email = body.get("email", "")
        username = email.split("@")[0]
    except json.JSONDecodeError:
        return error_response(400, 'Invalid JSON in request body')

    try:
        # Get system prompt for chatbot
        print("Getting system prompt for chatbot")
        response = s3_client.get_object(Bucket=PROMPT_BUCKET,
                                        Key=CHATBOT_PROMPT)
        chatbot_prompt = response['Body'].read().decode('utf-8')
    except Exception as e:
        print(f"Unexpected error while retrieving chatbot prompt: {e}")
        return error_response(500, str(e))

        # Get relevant election law chunks from KB
        # election_laws = retrieve_laws(user_query=user_query)
    try:
        # Get insights for candidate
        print("Getting insights for candidate")
        response = main_table.get_item(Key={
            'userId': f"USER#{email}",
            'SK': "INSIGHTS"
        })
        user_insights = response.get("Item")
    except Exception as e:
        print(f"Unexpected error while retrieving insights: {e}")
        return error_response(500, str(e))

    try:
        # Get questionnaire for candidate
        print("Getting questionnaire for candidate")
        response = questionnaire_table.get_item(Key={
            'userId': f"USER#{email}"
        })
        item = response.get("Item")
        questionnaire_answers = {k: v for k, v in item.items() if k not in IGNORED_FIELDS}

        questionnaire_text = prepare_user_context(questionnaire=questionnaire_answers)

    except Exception as e:
        print(f"Unexpected error while retrieving questionnaire: {e}")
        return error_response(500, str(e))

    try:
        # Fill system prompt
        print("Filling chatbot prompt")
        if questionnaire_text:
            print("Filling questionnaire")
            chatbot_prompt = chatbot_prompt.replace("{candidate_questionnaire}", questionnaire_text)
        if user_insights:
            print("Filling user insights")
            chatbot_prompt = chatbot_prompt.replace("{generated_insights}", user_insights)
        chatbot_prompt = chatbot_prompt.replace("{query}", str(user_query))
    except Exception as e:
        print(f"Unexpected error while filling chatbot prompt: {e}")
        return error_response(500, str(e))
        # chatbot_prompt = chatbot_prompt.replace("{relevant_election_laws}", election_laws)
        # Format query into message format.
        message = {
            'role': 'user',
            'content': [{'text': chatbot_prompt}]
        }

    try:
        # Load conversation history from DynamoDB instead of payload
        print(f"Loading conversation history from DynamoDB for chatId={chat_id}")
        conversation_history = load_conversation_history(chat_id)
    except Exception as e:
        print(f"Unexpected error while loading conversation history from DDB: {e}")
        return error_response(500, str(e))

        messages = conversation_history + [message]

    try:
        print("Converse call")
        response = bedrock_runtime.converse(
            modelId=MODEL_ID,
            messages=messages,
            inferenceConfig={
                'temperature': 0.3
            }
        )

        answer = response['output']['message']['content'][0]['text']

        # Write assistant message to DynamoDB
        print(f"Writing assistant message to DynamoDB for chatId={chat_id}")
        write_assistant_message(chat_id=chat_id, user_id=email, content=answer)

        # Use chatId-based S3 key for response
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
    except Exception as e:
        print(f"Unexpected error while running converse call: {e}")
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


def load_conversation_history(chat_id: str) -> list:
    """
    Query DynamoDB for all messages with the given chatId,
    sorted by timestamp ascending, excluding META records.
    Returns messages formatted for Bedrock converse API.
    """
    if not chat_history_table or not chat_id:
        return []

    from boto3.dynamodb.conditions import Key

    response = chat_history_table.query(
        KeyConditionExpression=Key("chatId").eq(chat_id),
        ScanIndexForward=True,  # ascending by sort key (timestamp)
    )

    messages = []
    for item in response.get("Items", []):
        # Exclude META records
        if item.get("timestamp") == "META":
            continue
        role = item.get("role", "user")
        content = item.get("content", "")
        messages.append({
            "role": role,
            "content": [{"text": content}],
        })

    return messages


def write_assistant_message(chat_id: str, user_id: str, content: str) -> None:
    """
    Write the assistant response message to DynamoDB.
    """
    if not chat_history_table:
        return

    now = datetime.now(timezone.utc).isoformat()
    chat_history_table.put_item(
        Item={
            "chatId": chat_id,
            "timestamp": now,
            "userId": user_id,
            "role": "assistant",
            "content": content,
        }
    )
        
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