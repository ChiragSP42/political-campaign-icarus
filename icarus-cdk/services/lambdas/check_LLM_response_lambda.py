import boto3
import json
import os
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource("dynamodb")

CHAT_HISTORY_TABLE = os.getenv("CHAT_HISTORY_TABLE", "")
chat_history_table = dynamodb.Table(CHAT_HISTORY_TABLE) if CHAT_HISTORY_TABLE else None


def lambda_handler(event, context):
    print(event)
    try:
        params = event.get('queryStringParameters', {})
        email = params.get("email", "")
        chat_id = params.get("chatId", "")
    except Exception as e:
        print(f"Failed to parse event: {e}")
        return error_response(400, {"status": "FAILED", "message": str(e)})

    if not chat_id:
        return error_response(400, {"status": "FAILED", "message": "chatId is required"})

    if not chat_history_table:
        return error_response(500, {"status": "FAILED", "message": "Chat history table not configured"})

    try:
        # Query DynamoDB for the most recent message in this chat (excluding META)
        response = chat_history_table.query(
            KeyConditionExpression=Key("chatId").eq(chat_id),
            ScanIndexForward=False,  # descending by timestamp
            Limit=5,
        )

        for item in response.get("Items", []):
            if item.get("timestamp") == "META":
                continue
            # The most recent non-META message determines the status:
            # If it's an assistant message, the response is ready.
            # If it's a user message, the bot hasn't responded yet.
            if item.get("role") == "assistant":
                return return_response(200, {
                    "status": "COMPLETED",
                    "message": item.get("content", "")
                })
            else:
                # Latest message is from user — still waiting for assistant
                return return_response(200, {
                    "status": "IN_PROGRESS",
                    "message": "LLM response not done yet"
                })

        # No messages at all
        return return_response(200, {
            "status": "IN_PROGRESS",
            "message": "LLM response not done yet"
        })

    except Exception as e:
        print(f"Error checking for response: {e}")
        return error_response(500, {"status": "FAILED", "message": str(e)})


def error_response(status_code, message: dict):
    return {
        'statusCode': status_code,
        'body': json.dumps(message),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }


def return_response(status_code, message: dict):
    return {
        'statusCode': status_code,
        'body': json.dumps(message),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }
