import boto3
import os
import json
from boto3.dynamodb.conditions import Key

# Initialize boto3 clients
dynamodb = boto3.resource("dynamodb")

# Environment variables
CHAT_HISTORY_TABLE = os.getenv("CHAT_HISTORY_TABLE", "")

# DynamoDB table reference
chat_history_table = dynamodb.Table(CHAT_HISTORY_TABLE) if CHAT_HISTORY_TABLE else None


def lambda_handler(event, context):
    """
    Session manager lambda for chat conversation history.
    Routes requests based on HTTP method and resource path.

    Operations:
        GET  /sessions          - List sessions for a user
        DELETE /sessions        - Delete a session and all its messages
        GET  /sessions/messages - Get messages for a session
    """
    try:
        http_method = event.get("httpMethod", "")
        resource = event.get("resource", "")
        params = event.get("queryStringParameters") or {}

        if resource == "/sessions/messages" and http_method == "GET":
            return get_messages(params)
        elif resource == "/sessions" and http_method == "GET":
            return list_sessions(params)
        elif resource == "/sessions" and http_method == "DELETE":
            return delete_session(params)
        else:
            return error_response(400, {"status": "FAILED", "message": f"Unsupported operation: {http_method} {resource}"})

    except Exception as e:
        print(f"Unexpected error: {e}")
        return error_response(500, {"status": "FAILED", "message": str(e)})


def list_sessions(params):
    """
    GET /sessions?userId=<email>
    Query userId-index GSI for META records, return sorted by createdAt descending.
    """
    user_id = params.get("userId", "")
    if not user_id:
        return error_response(400, {"status": "FAILED", "message": "Missing required parameter: userId"})

    if not chat_history_table:
        return error_response(500, {"status": "FAILED", "message": "Chat history table not configured"})

    response = chat_history_table.query(
        IndexName="userId-index",
        KeyConditionExpression=Key("userId").eq(user_id),
        ScanIndexForward=False,  # descending by createdAt
    )

    sessions = []
    for item in response.get("Items", []):
        # Only include META records (session metadata)
        if item.get("timestamp") != "META":
            continue
        sessions.append({
            "chatId": item.get("chatId", ""),
            "title": item.get("title", ""),
            "createdAt": item.get("createdAt", ""),
        })

    return return_response(200, sessions)


def delete_session(params):
    """
    DELETE /sessions?chatId=<uuid>&userId=<email>
    Query all records for chatId, batch-delete all. Return 404 if not found.
    """
    chat_id = params.get("chatId", "")
    user_id = params.get("userId", "")
    if not chat_id or not user_id:
        return error_response(400, {"status": "FAILED", "message": "Missing required parameters: chatId and userId"})

    if not chat_history_table:
        return error_response(500, {"status": "FAILED", "message": "Chat history table not configured"})

    # Query all records for this chatId
    response = chat_history_table.query(
        KeyConditionExpression=Key("chatId").eq(chat_id),
    )

    items = response.get("Items", [])
    if not items:
        return error_response(404, {"status": "FAILED", "message": "Session not found"})

    # Batch delete all records
    with chat_history_table.batch_writer() as batch:
        for item in items:
            batch.delete_item(
                Key={
                    "chatId": item["chatId"],
                    "timestamp": item["timestamp"],
                }
            )

    return return_response(200, {"status": "DELETED", "chatId": chat_id})


def get_messages(params):
    """
    GET /sessions/messages?chatId=<uuid>
    Query by chatId, exclude META, return messages sorted by timestamp ascending.
    """
    chat_id = params.get("chatId", "")
    if not chat_id:
        return error_response(400, {"status": "FAILED", "message": "Missing required parameter: chatId"})

    if not chat_history_table:
        return error_response(500, {"status": "FAILED", "message": "Chat history table not configured"})

    response = chat_history_table.query(
        KeyConditionExpression=Key("chatId").eq(chat_id),
        ScanIndexForward=True,  # ascending by timestamp
    )

    messages = []
    for item in response.get("Items", []):
        # Exclude META records
        if item.get("timestamp") == "META":
            continue
        messages.append({
            "role": item.get("role", ""),
            "content": item.get("content", ""),
            "timestamp": item.get("timestamp", ""),
        })

    return return_response(200, messages)


def error_response(status_code, message):
    """Helper function to return error responses."""
    return {
        "statusCode": status_code,
        "body": json.dumps(message),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }


def return_response(status_code, message):
    """Helper function to return success responses."""
    return {
        "statusCode": status_code,
        "body": json.dumps(message),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }
