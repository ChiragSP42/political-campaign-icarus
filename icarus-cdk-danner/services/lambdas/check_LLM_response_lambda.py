import boto3
import json
import os
import time
from botocore.config import Config

# Initialize Boto3 Clients
config = Config(
    read_timeout=300,
    connect_timeout=60,
    retries={
        'total_max_attempts': 5,
        'mode': 'adaptive'
    }
)

s3_client = boto3.client("s3")
sts_client = boto3.client("sts")

# Environment variables
ACCOUNT_ID = sts_client.get_caller_identity()['Account']
S3_RESPONSES = os.getenv("S3_RESPONSES", 'chatbot-responses')
S3_RESPONSES = f"{S3_RESPONSES}-{ACCOUNT_ID}"

def lambda_handler(event, context):
    print(event)
    print("Loading event and body...")
    try:
        body = event.get('queryStringParameters', '{}')
        print(f"Body: {body}")
        email = body.get("email", "")
        username = email.split("@")[0]
    except Exception as e:
        print("Failed to load body from events")
        print(e)

        message = {
            'status': "FAILED",
            'message': f"{e}"
        }
        return error_response(400, message)
    
    print("Starting to check if response has been generated...")
    try:
        response = s3_client.get_object(Bucket=S3_RESPONSES,
                                Key=f"{username}/{username}_response.md")
        print("Got response")
        chatbot_response = response['Body'].read().decode('utf-8')

        message = {
            'status': 'COMPLETED',
            'message': chatbot_response
        }
        return return_response(200, message)
    
    except s3_client.exceptions.NoSuchKey:
        print("LLM response not done yet")

        message = {
            'status': "IN_PROGRESS",
            'message': "LLM response not done yet"
        }
        return return_response(200, message)
    
    except Exception as e:
        print("Failed")
        message = {
            "status": "FAILED",
            "message": e
        }
    return error_response(400, message)

            
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
    """Helper function to return error responses."""
    return {
        'statusCode': status_code,
        'body': json.dumps(message),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }