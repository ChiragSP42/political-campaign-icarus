"""
Constants and configuration for Project Icarus Streamlit app
"""

import os
from dotenv import load_dotenv

load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Cognito Configuration
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID")
COGNITO_REGION = os.getenv("COGNITO_REGION", "us-east-1")

# API Configuration
API_ENDPOINT = os.getenv("API_ENDPOINT", '')
API_TIMEOUT = 60  # seconds

# Application Configuration
APP_NAME = os.getenv("APP_NAME", "Project Icarus")
APP_VERSION = os.getenv("APP_VERSION", "1.0")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Questionnaire Configuration
OFFICE_OPTIONS = [
    "House_of_Delegates",
    "U.S._House",
    "Congress",
    "Governor",
    "Local_Office",
    "President",
    "U.S._Senate",
    "Lieutenant_Governor",
    "Attorney_General",
    "Senate_of_Virginia",
    "Commonwealth's_Attorney",
    "Commissioner_of_the_Revenue",
    "County_Board_Member",
    "Sheriff",
    "Treasurer",
    "Clerk_of_Court",
    "School_Board",
    "Soil_and_Water_Conservation_Director",
    "Mayor",
    "City_Council",
    "Town_Council",
    "Board_of_Supervisors"
]

BACKGROUND_QUESTIONS = [
    "militaryBackground",
    "publicSafety",
    "unionBackground",
    "businessOwner",
    "publicService",
    "faithCommunity",
    "firstCampaign"
]

ARCHETYPE_QUESTIONS = [
    "themeSong",
    "debateReaction",
    "coffeeShopIntro",
    "roleModel",
    "preferredEvent",
    "decisionStyle",
    "tagline",
    "socialMedia",
    "opponentResponse",
    "symbolism",
    "headline"
]

# Session Configuration
SESSION_TIMEOUT = 3600  # 1 hour in seconds
MAX_CHAT_HISTORY = 50   # Max messages to keep in memory
