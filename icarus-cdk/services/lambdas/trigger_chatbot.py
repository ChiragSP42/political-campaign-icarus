import boto3
import os
import json
import uuid
from datetime import datetime, timezone

# Initialize boto3 clients
lambda_client = boto3.client("lambda")
dynamodb = boto3.resource("dynamodb")

# Environment variables
CHATBOT_LAMBDA_NAME = os.getenv("CHABOT_LAMBDA_NAME", 'chatbot-lambda')
CHAT_HISTORY_TABLE = os.getenv("CHAT_HISTORY_TABLE", "")

# DynamoDB table reference
chat_history_table = dynamodb.Table(CHAT_HISTORY_TABLE) if CHAT_HISTORY_TABLE else None

def lambda_handler(event, context):
    """
    Function to start async process of generating chatbot response by triggering chatbot lambda.
    Persists user messages to DynamoDB and manages chat session creation.

    Args:
        event (Dict): Event object
        context (_type_): _description_

    Returns:
        Dict: JSON response of format

            {
                'statusCode': status_code,
                'body': {
                        'status': COMPLETED|FAILED,
                        'message': <any message or content>,
                        'chatId': <uuid string>
                    },
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
        }
    """
    try:
        print("Parsing event object")
        # Parse the request body
        body = json.loads(event.get('body', {}))
        print(f"Body: {body}")
        user_query = body.get("query", "")
        email = body.get("email", "")
        username = email.split("@")[0]
        chat_id = body.get("chatId", None)

    except Exception as e:
        print(f"Failed to parse event object: {e}")
        message = {
            'status': 'FAILED',
            'message': f"{e}"
        }
        return error_response(status_code=400, message=message)

    try:
        # Generate chatId if not provided (new session)
        is_new_session = chat_id is None
        if is_new_session:
            chat_id = str(uuid.uuid4())

        now = datetime.now(timezone.utc).isoformat()

        # Write META record for new sessions
        if is_new_session and chat_history_table:
            title = user_query[:50]
            print(f"Creating new session META record for chatId={chat_id}")
            chat_history_table.put_item(
                Item={
                    "chatId": chat_id,
                    "timestamp": "META",
                    "userId": email,
                    "createdAt": now,
                    "title": title,
                }
            )

        # Write user message record
        if chat_history_table:
            print(f"Writing user message to DynamoDB for chatId={chat_id}")
            chat_history_table.put_item(
                Item={
                    "chatId": chat_id,
                    "timestamp": now,
                    "userId": email,
                    "role": "user",
                    "content": user_query,
                }
            )

        print("Invoking chatbot lambda")
        lambda_client.invoke(
            FunctionName=CHATBOT_LAMBDA_NAME,
            InvocationType='Event',
            Payload=json.dumps({
                'body': {
                    "query": user_query,
                    "chatId": chat_id,
                    "email": email
                }
            })
        )
        print("Successfully done")

        message = {
            'status': 'COMPLETED',
            'message': "Triggered successfully",
            'chatId': chat_id
        }
        return return_response(status_code=200, message=message)
    except Exception as e:
        print(f"Failed to invoke lambda: {e}")
        message = {
            'status': 'FAILED',
            'message': f"{e}"
        }
        return error_response(status_code=400, message=message)


def error_response(status_code, message: dict):
    """Helper function to return error responses."""
    return {
        'statusCode': status_code,
        'body': json.dumps(message),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }

def return_response(status_code, message: dict):
    """Helper function to return success responses."""
    return {
        'statusCode': status_code,
        'body': json.dumps(message),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }
