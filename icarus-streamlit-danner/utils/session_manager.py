"""
Session state management for Streamlit app
"""

import streamlit as st
import boto3
from datetime import datetime
from typing import Optional, Dict, Any, List

# Initialize S3 client
s3_client = boto3.client('s3')
sts_client = boto3.client("sts")
ACCOUNT_ID = sts_client.get_caller_identity()['Account']
S3_BUCKET_INSIGHTS = 'generated-insights'  # Your insights bucket name
S3_BUCKET_INSIGHTS = f"{S3_BUCKET_INSIGHTS}-{ACCOUNT_ID}"
S3_QUESTIONNAIRES = 'icarus-questionnaires'
S3_QUESTIONNAIRES = f"{S3_QUESTIONNAIRES}-{ACCOUNT_ID}"

class SessionManager:
    """Manages Streamlit session state"""
    
    @staticmethod
    def initialize_session():
        """Initialize session variables if they don't exist"""
        
        if 'user_email' not in st.session_state:
            st.session_state.user_email = None
            
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
            
        if 'questionnaire_completed' not in st.session_state:
            st.session_state.questionnaire_completed = False
            
        if 'questionnaire_data' not in st.session_state:
            st.session_state.questionnaire_data = {}
        
        if 'signup_email' not in st.session_state:
            st.session_state.signup_email = ''
            
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
            
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "auth"
            
        if 'session_start_time' not in st.session_state:
            st.session_state.session_start_time = datetime.now()

        if 'current_page' not in st.session_state:
            st.session_state.current_page = "auth"  # default

    @staticmethod
    def set_user(email: str, questionnaire_completed: bool = False):
        """Set authenticated user"""
        st.session_state.user_email = email
        st.session_state.authenticated = True
        st.session_state.questionnaire_completed = questionnaire_completed
    
    @staticmethod
    def get_user() -> Optional[str]:
        """Get current user email"""
        return st.session_state.get('user_email')
    
    @staticmethod
    def is_authenticated() -> bool:
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)
    
    @staticmethod
    def logout():
        """Logout user"""
        st.session_state.user_email = None
        st.session_state.authenticated = False
        st.session_state.questionnaire_completed = False
        st.session_state.questionnaire_data = {}
        st.session_state.chat_messages = []
    
    @staticmethod
    def add_chat_message(role: str, content: str):
        """Add message to chat history"""
        st.session_state.chat_messages.append({
            "role": role,
            "content": [{"text": content}]
        })
    
    @staticmethod
    def get_chat_history() -> List[Dict[str, str]]:
        """Get formatted chat history for API"""
        messages = []
        for msg in st.session_state.chat_messages:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return messages
    
    @staticmethod
    def set_questionnaire_data(data: Dict[str, Any]):
        """Save questionnaire data"""
        st.session_state.questionnaire_data = data
        st.session_state.questionnaire_completed = True
    
    @staticmethod
    def check_questionnaire_exists() -> bool:
        response = False
        email = SessionManager.get_user()
        if email:
            username = email.split("@")[0]
            questionnaire_key = f"{username}/{username}_questionnaire.json"
            response = s3_client.get_object(
                Bucket=S3_QUESTIONNAIRES,
                Key=questionnaire_key
            )
        if not response:
            return False
        else:
            return True
