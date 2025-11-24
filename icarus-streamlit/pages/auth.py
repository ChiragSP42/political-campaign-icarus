"""
Authentication page for sign up and sign in
"""

import streamlit as st
from utils.auth_manager import AuthManager
from utils.session_manager import SessionManager
from utils.api_client import APIClient

st.set_page_config(
    page_title="Project Icarus - Auth",
    page_icon="🎭",
    layout="centered"
)

# Initialize session
SessionManager.initialize_session()

# Hide sidebar on auth page
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown("""
        <h1 style='text-align: center; color: #667eea;'>
            🎭 Project Icarus
        </h1>
        <h3 style='text-align: center; color: #666;'>
            Campaign Strategy AI
        </h3>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Tab selection
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    
    auth_manager = AuthManager()
    api_client = APIClient()
    
    with tab1:
        st.subheader("Sign In")
        
        with st.form("signin_form"):
            email = st.text_input(
                "Email",
                placeholder="your.email@example.com"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )
            
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    with st.spinner("Signing in..."):
                        success, message = auth_manager.sign_in(email, password)
                        
                        if success:
                            # Check if questionnaire exists
                            check_result = api_client.check_questionnaire_exists(email)
                            questionnaire_completed = check_result.get("exists", False)
                            
                            # Set user in session
                            SessionManager.set_user(email, questionnaire_completed)
                            
                            st.success(f"Signed in as {email}")
                            
                            # Redirect to appropriate page
                            if questionnaire_completed:
                                st.session_state.current_page = "chat"
                                st.switch_page("pages/chatbot.py")
                            else:
                                print("Questionnaire not filled")
                                st.session_state.current_page = "questionnaire"
                                st.switch_page("pages/questionnaire.py")
                            
                            # st.rerun()
                        else:
                            st.error(message)
    
    with tab2:
        st.subheader("Create Account")
        
        # Step 1: Sign Up
        if 'signup_step' not in st.session_state:
            st.session_state.signup_step = 1
        
        if st.session_state.signup_step == 1:
            with st.form("signup_form"):
                email = st.text_input(
                    "Email",
                    placeholder="your.email@example.com",
                    key="signup_email"
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="8+ chars, uppercase, lowercase, numbers",
                    key="signup_password"
                )
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    key="signup_confirm"
                )
                
                submit = st.form_submit_button("Sign Up", use_container_width=True)
                
                if submit:
                    if not email or not password or not confirm_password:
                        st.error("Please fill in all fields")
                    elif password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        with st.spinner("Creating account..."):
                            success, message = auth_manager.sign_up(email, password)
                            
                            if success:
                                st.session_state.signup_step = 2
                                st.session_state.signup_email_verified = email
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
        
        elif st.session_state.signup_step == 2:
            st.info(f"Verify your email: {st.session_state.signup_email}")
            st.write("Enter the verification code sent to your email:")
            
            with st.form("verify_form"):
                code = st.text_input(
                    "Verification Code",
                    placeholder="6-digit code",
                    key="verification_code"
                )
                
                submit = st.form_submit_button("Verify Email", use_container_width=True)
                
                if submit:
                    if not code:
                        st.error("Please enter verification code")
                    else:
                        with st.spinner("Verifying..."):
                            success, message = auth_manager.confirm_sign_up(
                                st.session_state.signup_email_verified,
                                code
                            )
                            
                            if success:
                                st.session_state.signup_step = 1
                                st.success(message)
                                st.info("Switching to sign in tab...")
                                st.rerun()
                            else:
                                st.error(message)

if __name__ == "__main__":
    main()
