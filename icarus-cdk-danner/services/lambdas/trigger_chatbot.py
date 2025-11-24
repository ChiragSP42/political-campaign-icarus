import boto3
import os
import json

# Initialize boto3 clients
lambda_client = boto3.client("lambda")

# Environment variables
CHATBOT_LAMBDA_NAME = os.getenv("CHABOT_LAMBDA_NAME", 'chatbot-lambda')

def lambda_handler(event, context):
    """
    Function to start async process of generating chatbot response by triggering chatbot lambda

    Args:
        event (Dict): Event object
        context (_type_): _description_

    Returns:
        Dict: JSON response of format

            {
                'statusCode': status_code,
                'body': {
                        'status': COMPLETED|FAILED,
                        'message': <any message or content>
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
        conversation_history = body.get("conversation_history", [])
        email = body.get("email", "")
        username = email.split("@")[0]

    except Exception as e:
        print(f"Failed to parse event object: {e}")
        message = {
            'status': 'FAILED',
            'message': f"{e}"
        }
        return error_response(status_code=400, message=message)
    
    try:
        print("Invoking chatbot lambda")
        lambda_client.invoke(
            FunctionName=CHATBOT_LAMBDA_NAME,
            InvocationType='Event',
            Payload=json.dumps({
                'body': {
                    "query": user_query,
                    "conversation_history": conversation_history,
                    "email": email
                }
            })
        )
        print("Successfully done")

        message = {
            'status': 'COMPLETED',
            'message': "Triggered successfully"
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
    """Helper function to return error responses."""
    return {
        'statusCode': status_code,
        'body': json.dumps(message),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }