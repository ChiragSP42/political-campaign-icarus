"""
Lambda Function: check-questionnaire

Purpose:
  Check if a user has already completed the questionnaire in DynamoDB.
  
Input:
  - Query parameter: email (user's email address)
  
Output:
  - {exists: true/false} indicating if questionnaire exists
  
How it works:
  1. Extract email from query parameters
  2. Query DynamoDB icarus-questionnaires table
  3. Return whether item exists
  
Why this matters:
  - After sign-in, we need to know if user should go to questionnaire or chat
  - This makes the routing decision
"""

import json
import boto3
import os
from botocore.exceptions import ClientError

# Initialize S3 resource
s3_client = boto3.client("s3")
sts_client = boto3.client("sts")

ACCOUNT_ID = sts_client.get_caller_identity()['Account']
S3_QUESTIONNAIRES = os.getenv("S3_QUESTIONNAIRES", 'icarus-questionnaires')
S3_QUESTIONNAIRES = f"{S3_QUESTIONNAIRES}-{ACCOUNT_ID}"

def lambda_handler(event, context):
    """
    Main Lambda handler function.
    
    AWS calls this when API Gateway receives a request.

    Event format: {'email': The email ID}
    """
    
    try:
        print(event)
        body = event.get('queryStringParameters', '{}')
        print(f"Loaded body: {body}")
        email = body.get('email') if body else None
        print(f"Extracted email: {email}")
        if not email:
          return {
              'statusCode': 400,
              'body': json.dumps({'error': 'Email parameter is required'}),
              'headers': {
                  'Content-Type': 'application/json',
                  'Access-Control-Allow-Origin': '*'
              }
          }
        
        # Query S3
        username = email.split("@")[0]
        print("Getting questionnaire")
        response = s3_client.get_object(Bucket=S3_QUESTIONNAIRES,
                                        Key=f"{username}/{username}_questionnaire.json")
        if response:
            print("Sending back")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'exists': True,
                    'email': email
                }),
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
            }
    
    except Exception as e:
        # Unexpected error
        print(f"Unexpected error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }