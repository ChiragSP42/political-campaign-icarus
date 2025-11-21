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
DEBUG = os.getenv("DEBUG", "False").lower() == "True"

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

# Election cycles
ELECTION_CYCLE = {
    "elections": {
        "President": {
                "id": 1,
                "cycle": 4,
                "election_pattern": "even",
                "is_statewide": True,
                "description": "Presidential elections occur every 4 years in even-numbered years"
            },
        "U.S._Senate": {
            "id": 6,
            "cycle": 6,
            "election_pattern": "even_biennial",
            "is_statewide": True,
            "description": "Senate elections occur every 6 years, with approximately 1/3 of seats up every 2 years"
        },
        "U.S._House": {
            "id": 5,
            "cycle": 2,
            "election_pattern": "even",
            "is_statewide": False,
            "description": "House elections occur every 2 years in even-numbered years"
        },
        "Governor": {
            "id": 3,
            "cycle": 4,
            "election_pattern": "odd",
            "is_statewide": True,
            "description": "Gubernatorial elections occur every 4 years in odd-numbered years (Virginia)"
        },
        "Lieutenant_Governor": {
            "id": 4,
            "cycle": 4,
            "election_pattern": "odd",
            "is_statewide": True,
            "description": "Lieutenant Governor elections occur every 4 years in odd-numbered years (Virginia)"
        },
        "Attorney_General": {
            "id": 12,
            "cycle": 4,
            "election_pattern": "odd",
            "is_statewide": True,
            "description": "Attorney General elections occur every 4 years in odd-numbered years (Virginia)"
        },
        "Senate_of_Virginia": {
            "id": 9,
            "cycle": 4,
            "election_pattern": "odd",
            "is_statewide": False,
            "description": "Virginia State Senate elections occur every 4 years in odd-numbered years"
        },
        "House_of_Delegates": {
            "id": 8,
            "cycle": 2,
            "election_pattern": "odd",
            "is_statewide": False,
            "description": "House of Delegates elections occur every 2 years in odd-numbered years (Virginia)"
        },
        "Commonwealth's_Attorney": {
            "id": 530,
            "cycle": 4,
            "election_pattern": "odd",
            "is_statewide": False,
            "description": "Commonwealth's Attorney elections occur every 4 years in odd-numbered years (Virginia)"
        },
        "Commissioner_of_the_Revenue": {
            "id": 552,
            "cycle": 4,
            "election_pattern": "odd",
            "is_statewide": False,
            "description": "Commissioner of the Revenue elections occur every 4 years in odd-numbered years (Virginia)"
        },
        "County_Board_Member": {
            "id": 546,
            "cycle": 4,
            "election_pattern": "odd",
            "is_statewide": False,
            "description": "County_Board_Member elections occur every 4 years in odd-numbered years (Virginia)"
        },
        "Sheriff": {
            "id": 386,
            "cycle": 4,
            "election_pattern": "odd",
            "is_statewide": False,
            "description": "Sheriff elections occur every 4 years in odd-numbered years (Virginia)"
        },
        "Treasurer": {
            "id": 389,
            "cycle": 4,
            "election_pattern": "odd",
            "is_statewide": False,
            "description": "Treasurer elections occur every 4 years in odd-numbered years (Virginia)"
        },
        "Clerk_of_Court": {
            "id": 545,
            "cycle": 8,
            "election_pattern": "odd",
            "is_statewide": False,
            "description": "Clerk of Court elections occur every 8 years in odd-numbered years (Virginia)"
        },
        "School_Board": {
            "id": 549,
            "cycle": 4,
            "election_pattern": "periodic",
            "is_statewide": False,
            "description": "School Board elections in Virginia occur every 4 years, 2 years, or annually depending on the jurisdiction"
        },
        "Soil_and_Water_Conservation_Director": {
            "id": 553,
            "cycle": 4,
            "election_pattern": "odd",
            "is_statewide": False,
            "description": "Soil and Water Conservation Director elections occur every 4 years in odd-numbered years (Virginia)"
        },
        "Mayor": { 
            "id": 73,
            "cycle": 4,
            "election_pattern": "periodic",
            "is_statewide": False,
            "description": "Mayoral elections vary by jurisdiction, typically 2-4 year terms"
        },
        "City_Council": {
            "id": 551,
            "cycle": 4,
            "election_pattern": "periodic",
            "is_statewide": False,
            "description": "City Council elections typically occur every 4 years with staggered terms (Virginia)"
        },
        "Town_Council": {
            "id": 547,
            "cycle": 4,
            "election_pattern": "periodic",
            "is_statewide": False,
            "description": "Town Council elections typically occur every 4 years with staggered terms"
        },
        "Board_of_Supervisors": {
            "id": 419,
            "cycle": 4,
            "election_pattern": "odd",
            "is_statewide": False,
            "description": "Board of Supervisors elections occur every 4 years (often staggered)"
        }
    }
}