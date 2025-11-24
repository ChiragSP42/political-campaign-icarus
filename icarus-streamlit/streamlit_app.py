"""
Main entry point for Project Icarus Streamlit application
"""

import streamlit as st
from utils.session_manager import SessionManager

st.set_page_config(
    page_title="Project Icarus",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session
SessionManager.initialize_session()

def main():
    # Check authentication
    if SessionManager.is_authenticated():
        print("Authenticated")
        # Check questionnaire
        if SessionManager.check_questionnaire_exists():
            # Redirect to chat
            print("Redirecting to chat window")
            st.switch_page("pages/chatbot.py")
        else:
            # Redirect to questionnaire
            print("Redirecting to questionnaire window")
            st.switch_page("pages/questionnaire.py")
    else:
        # Show auth page
        print("Get authenticated")
        st.switch_page("pages/auth.py")
    
    st.rerun()

if __name__ == "__main__":
    main()
