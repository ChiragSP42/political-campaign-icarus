import os
from aws_helpers import helpers
import boto3
from dotenv import load_dotenv
load_dotenv(override=True)

aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")

session = boto3.Session(aws_access_key_id=aws_access_key,
                        aws_secret_access_key=aws_secret_key,
                        region_name='us-east-1')

bedrock_agent = session.client("bedrock-runtime")
# MODEL_ID = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
MODEL_ID = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'

with open("content_retrieval_prompt.txt", 'r') as f:
    content_retrieval_prompt = f.read()

# user_prompt = 'Office position: House of Delegates\nYear: 2025\n\nWhat would be my message strategy for District 5?'
# user_prompt = 'Office position: House of Delegates\nYear: 2025\n\nWhat are the precincts with the lowest flip numbers?'
user_prompt = 'Office position: House of Delegates\nYear: 2025\n\nWhat does the trend look like for democrats to win the governor (or other) elections?'

message = f"{content_retrieval_prompt}\n\nUser prompt: {user_prompt}"

messages = [{
    "role": "user",
    "content": [
        {
            "text": message
        }
    ]
}]

response = bedrock_agent.converse(modelId=MODEL_ID,
                                  messages=messages)

output_message = response['output']['message']
final_answer = output_message['content'][0]['text']
print(f"Query:\n{user_prompt}")
print (final_answer)