"""
Lambda Function: save-questionnaire

Purpose:
  Store questionnaire answers in DynamoDB for future retrieval.
  
Input (JSON):
  {
    "email": "user@example.com",
    "answers": {
      "fullName": "John Doe",
      "office": "House_of_Delegates",
      ...all questionnaire answers
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }
  
Output:
  - {success: true, message: "Questionnaire saved"}
  
Why this matters:
  - This is where candidate's profile is stored
  - Later, the chatbot Lambda will retrieve this to give AI context
  - The AI uses these answers to determine messaging strategy
"""

import json
import boto3
import os
from datetime import datetime
from botocore.exceptions import ClientError

# Initialize S3 resource
# s3_client = boto3.client("s3")
# sts_client = boto3.client("sts")
dynamodb = boto3.resource('dynamodb')

# ACCOUNT_ID = sts_client.get_caller_identity()['Account']
# S3_QUESTIONNAIRES = os.getenv("S3_QUESTIONNAIRES", 'icarus-questionnaires')
# S3_QUESTIONNAIRES = f"{S3_QUESTIONNAIRES}-{ACCOUNT_ID}"
QUESTIONNAIRE_TABLE_NAME = os.getenv("QUESTIONNAIRE_TABLE_NAME")

questionnaire_table = dynamodb.Table(QUESTIONNAIRE_TABLE_NAME)


def lambda_handler(event, context):
    """
    Main Lambda handler function for saving questionnaire.

    The event format: {'email': The email ID, 'answers': The questionnaire answers}
    """
    
    try:
        # Parse the request body
        body = json.loads(event.get('body', '{}'))
        
        email = body.get('email')
        answers = body.get('answers')
        timestamp = body.get('timestamp', datetime.now().isoformat())
        
        # Validate inputs
        if not email:
            return error_response(400, 'Email is required')
        
        if not answers:
            return error_response(400, 'Answers are required')
        
        username = email.split("@")[0]

        try:
            # Save to DDB
            try:
                item = {
                    'userId': f"USER#{email}",
                    'SK': f"META#QUESTIONNAIRE",
                    'savedAt': timestamp,
                    'updatedAt': datetime.now().isoformat(),
                }
                item = item | answers if isinstance(answers, dict) else item | json.loads(answers)
                print("Item to be saved: ", item)
                questionnaire_table.put_item(Item=item)
            except Exception as e:
                print("Failed to save in DDB")
        except:
            print("Failed to save questionnaire")
        
        print(f"Saved questionnaire for {username}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'message': 'Questionnaire saved successfully',
                'email': email,
                'savedAt': timestamp
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
    
    except json.JSONDecodeError:
        return error_response(400, 'Invalid JSON in request body')
    
    except ClientError as e:
        print(f"DynamoDB error: {e}")
        return error_response(500, 'Failed to save questionnaire')
    
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