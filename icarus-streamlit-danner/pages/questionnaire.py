"""
Updated questionnaire.py - Step 4 with S3 Polling

This adds a waiting/processing step after submission that polls S3 
for the generated insights file before redirecting to chat.
"""

import streamlit as st
import boto3
import time
from utils.session_manager import SessionManager
from utils.api_client import APIClient
from utils.constants import OFFICE_OPTIONS, ELECTION_CYCLE

st.set_page_config(
    page_title="Project Icarus - Questionnaire",
    page_icon="📋",
    layout="wide"
)

SessionManager.initialize_session()

if not SessionManager.is_authenticated():
    st.error("Please sign in first")
    st.stop()

# Initialize S3 client
s3_client = boto3.client('s3')
sts_client = boto3.client("sts")
ACCOUNT_ID = sts_client.get_caller_identity()['Account']
S3_BUCKET_INSIGHTS = 'generated-insights'  # Your insights bucket name
S3_BUCKET_INSIGHTS = f"{S3_BUCKET_INSIGHTS}-{ACCOUNT_ID}"

def check_insights_file_exists(email):
    """
    Check if insights file exists in S3
    Returns: (exists: bool, file_content: str or None)
    """
    try:
        username = email.split("@")[0]
        insights_key = f"{username}/{username}_insights.md"
        response = s3_client.get_object(
            Bucket=S3_BUCKET_INSIGHTS,
            Key=insights_key
        )
        content = response['Body'].read().decode('utf-8')
        return True, content
    except s3_client.exceptions.NoSuchKey:
        return False, None
    except s3_client.exceptions.InvalidObjectState:
        print("Invalid object state")
        return False, None
    except Exception as e:
        print(f"Error checking insights file: {e}")
        return False, None
    
def delete_old_insights(email):
    """
    Check if insights file exists in S3
    Returns: (exists: bool, file_content: str or None)
    """
    try:
        username = email.split("@")[0]
        insights_key = f"{username}/{username}_insights.md"
        response = s3_client.delete_object(
            Bucket=S3_BUCKET_INSIGHTS,
            Key=insights_key
        )
        content = response['Body'].read().decode('utf-8')
        return True, content
    except s3_client.exceptions.NoSuchKey:
        return False, None
    except Exception as e:
        print(f"Error checking insights file: {e}")
        return False, None

def main():
    st.title("📋 Candidate Intake Questionnaire")
    st.write("Help us understand your campaign better")

    # Initialize step
    if 'questionnaire_step' not in st.session_state:
        st.session_state.questionnaire_step = 1
        st.session_state.form_data = {}

    # Step 1: Basic Information
    if st.session_state.questionnaire_step == 1:
        with st.container():
            st.subheader("Step 1: Basic Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                full_name = st.text_input(
                    "Full Name *",
                    key="full_name",
                    value=st.session_state.form_data.get("fullName", "")
                )
            
            with col2:
                office = st.selectbox(
                    "Office Running For *",
                    OFFICE_OPTIONS,
                    key="office",
                    index=OFFICE_OPTIONS.index(st.session_state.form_data.get("office", "House_of_Delegates"))
                    if st.session_state.form_data.get("office") in OFFICE_OPTIONS else 0
                )
            
            elections = ELECTION_CYCLE['elections']
            election = elections.get(office, {})
            if election.get('is_statewide') == True:
                district = st.selectbox(
                    "District Running In (or Statewide) *",
                    ['Statewide'],
                    key="statewide",
                    index=0
                )
            else:
                district = st.selectbox(
                    "District Running In (or Statewide) *",
                    [f"District_{i}" for i in range(1, 101)],
                    key='district',
                    index=0
                )
            
            # Save to session
            st.session_state.form_data["fullName"] = full_name
            st.session_state.form_data["office_position"] = office
            st.session_state.form_data["district_name"] = district
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Next →", use_container_width=True):
                    if not full_name or not office or not district:
                        st.error("Please fill in all required fields")
                    else:
                        st.session_state.questionnaire_step = 2
                        st.rerun()

    # Step 2: Background & Profile
    elif st.session_state.questionnaire_step == 2:
        st.subheader("Step 2: Background & Profile")
        st.write("Tell us about your background and credibility anchors")
        
        background_questions = {
            "militaryBackground": "Do you have a military background (veteran, active duty, or immediate family)?",
            "publicSafety": "Have you served in law enforcement, firefighting, or another public safety role?",
            "unionBackground": "Do you come from a union household or have direct ties to labor/organizing?",
            "businessOwner": "Are you a small business owner or entrepreneur?",
            "publicService": "Have you held any public service roles (school board, council, community board)?",
            "faithCommunity": "Do you identify strongly with a faith community or civic organization?",
            "firstTime": "Is this your first campaign, or have you run for office before?"
        }
        
        for key, question in background_questions.items():
            st.session_state.form_data[question] = st.checkbox(
                question,
                value=st.session_state.form_data.get(key, False),
                key=f"bg_{key}"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous", use_container_width=True):
                st.session_state.questionnaire_step = 1
                st.rerun()
        with col2:
            if st.button("Next →", use_container_width=True):
                st.session_state.questionnaire_step = 3
                st.rerun()

    # Step 3: Communication Style
    elif st.session_state.questionnaire_step == 3:
        st.subheader("Step 3: Communication Style & Archetype")
        st.write("Help us understand your communication style")
        
        archetype_questions = {
            "themeSong": ("If your campaign had a theme song, what would it sound like?", [
                "🎶 Upbeat pop anthem – optimistic, inclusive, energetic",
                "🎸 Rock/hip-hop banger – fiery, bold, disruptive",
                "🎻 Folk/acoustic ballad – personal, grounded, community-focused",
                "🎺 Patriotic or orchestral march – traditional, serious, trustworthy",
                "🎷 Jazz / soulful groove – creative, improvisational, approachable",
                "🎤 Country anthem – rooted, local, values-driven",
                "🎧 Electronic / techno beat – modern, youthful, future-focused",
                "🎹 Piano/classical piece – thoughtful, steady, intellectual",
            ]),
            "debateReaction": ("At your first debate, you get a tough, unexpected question. What do you do?", [
                "Answer honestly, even if imperfect",
                "Pivot to a key policy plan",
                "Use humor to break tension",
                "Respond passionately about values",
            ]),
            "coffeeShopIntro": ("When you meet voters in a coffee shop, how do you introduce yourself?", [
                "Ask them what issues matter most",
                "Introduce yourself formally with experience",
                "Make a friendly, neighborly comment first",
                "Dive straight into an issue",
            ]),
            "leadershipStyle": ("Which leader's style feels closest to yours?", [
                "Barack Obama (inspirational coalition-builder)",
                "Elizabeth Warren (policy fighter)",
                "Alexandria Ocasio-Cortez (energetic change agent)",
                "Joe Biden (empathetic unifier)"
            ]),
            "eventExcitement": ("Which event excites you most?", [
                "Town hall with live Q&A",
                "Big rally with cheering supporters",
                "Policy roundtable with experts",
                "Block party with neighbors"
            ]),
            "quickDecisions": ("When making a quick decision on policy, what guides you most?", [
                "Consensus with advisors",
                "Gut and values",
                "Data and evidence",
                "Constituents' views"
            ]),
            "tagLine": ("Which tagline would you pick for your campaign?", [
                "“For a Stronger, Fairer Community”",
                "“New Leadership, New Ideas”",
                "“Proven Experience. Trusted Results.”",
                "“Standing Up for Working Families”"
            ]),
            "socialMedia": ("On social media, how would you announce a new policy?", [
                "With facts and clarity (informative)",
                "Through a short story about someone affected",
                "With bold, catchy language (edgy)",
                "As an optimistic call to action"
            ]),
            "negativeComments": ("Your opponent airs a negative ad against you. How do you respond?", [
                "Calmly fact-check it",
                "Respond with a clever or story-driven video",
                "Hit back hard in your next speech",
                "Stay positive and continue with your message"
            ]),
            "Success": ("After the election, beyond winning, what would make you feel most successful?", [
                "Inspiring new voters and building a movement",
                "Shifting the community conversation",
                "Running with integrity and positivity",
                "Building lasting coalitions"
            ]),
            "Symbolism": ("If your campaign were represented by a symbol, which feels most like you?", [
                "A star ⭐️ - standing for integrity, recognition, and achievement.",
                "A bridge 🌉 - connecting people and ideas across divides.",
                "A flame 🔥 - energy, urgency, and fighting spirit.",
                "A toolbox 🧰 - practical, focused on solving problems.",
                "A sunrise 🌄 - hope, inspiration, and new beginnings."
            ]),
            "Headlines": ("Imagine the local newspaper writes a headline about your campaign. Which one would you prefer?", [
                "“Candidate Brings Community Together Across Divides”",
                "“Candidate Pushes Bold New Ideas to Shake Up the System”",
                "“Candidate Offers Detailed Plan to Fix Local Problems”",
                "“Candidate Inspires Hope for a Brighter Future”"
            ])
        }
        
        for key, (question, options) in archetype_questions.items():
            st.session_state.form_data[question] = st.radio(
                question,
                options,
                key=f"arch_{key}",
                index=options.index(st.session_state.form_data.get(key, options[0]))
                if st.session_state.form_data.get(key) in options else 0
            )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous", use_container_width=True):
                st.session_state.questionnaire_step = 2
                st.rerun()
        with col2:
            if st.button("Submit", use_container_width=True, type="primary"):
                # Submit questionnaire to API
                api_client = APIClient()
                email = SessionManager.get_user()
                
                with st.spinner("Saving your questionnaire..."):
                    response = api_client.save_questionnaire(
                        email=email,
                        answers=st.session_state.form_data
                    )
                
                if response.get("success"):
                    # Delete old insights file if exists
                    exists, content = check_insights_file_exists(email)
                
                    if exists and content:
                        delete_old_insights(email=email)
                    # Move to processing step
                    st.session_state.questionnaire_step = 4
                    st.session_state.poll_start_time = time.time()
                    st.rerun()
                else:
                    st.error(f"Failed to save questionnaire: {response.get('message', 'Unknown error')}")

    # Step 4: Processing & Waiting for Insights
    elif st.session_state.questionnaire_step == 4:
        st.subheader("🔄 Generating Your Campaign Insights")
        
        email = SessionManager.get_user()
        
        # Create a nice loading UI
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h3>📊 Analyzing Your Campaign Data</h3>
                <p>We're processing historical election data, precinct-level insights, 
                and generating personalized strategies for your campaign.</p>
                <p><strong>This typically takes 1-2 minutes.</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress indicator
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Polling mechanism
            max_wait_time = 180  # 3 minutes max
            poll_interval = 5  # Check every 5 seconds
            elapsed_time = 0
            
            if 'poll_start_time' not in st.session_state:
                st.session_state.poll_start_time = time.time()
            
            # Animated progress messages
            progress_messages = [
                "🔍 Analyzing historical election data...",
                "📈 Calculating Win Gap scenarios...",
                "🗺️ Processing precinct-level targeting...",
                "💡 Generating strategic recommendations...",
                "🎯 Finalizing your campaign insights..."
            ]
            
            # Start polling
            insights_ready = False
            insights_content = None
            
            while elapsed_time < max_wait_time:
                elapsed_time = time.time() - st.session_state.poll_start_time
                
                # Check if file exists
                exists, content = check_insights_file_exists(email)
                
                if exists and content:
                    insights_ready = True
                    insights_content = content
                    # Save to session
                    st.session_state.insights_content = content
                    st.session_state.questionnaire_completed = True
                    break
                
                # Update progress UI
                progress_percentage = min(int((elapsed_time / max_wait_time) * 100), 95)
                progress_placeholder.progress(progress_percentage)
                
                # Rotate through messages
                message_index = int(elapsed_time / 20) % len(progress_messages)
                status_placeholder.info(progress_messages[message_index])
                
                # Wait before next poll
                time.sleep(poll_interval)
                
                # Force Streamlit to update UI
                st.rerun()
            
            # Handle results
            if insights_ready:
                progress_placeholder.progress(100)
                status_placeholder.success("✅ Insights generated successfully!")
                st.balloons()
                time.sleep(1)
                
                # Navigate to chat page
                st.switch_page("pages/chatbot.py")
            else:
                status_placeholder.error("""
                    ⚠️ Insights generation is taking longer than expected.
                    
                    This could mean:
                    - Heavy processing load on our servers
                    - Large volume of historical data being analyzed
                    
                    Please wait a moment and click the button below to check again.
                """)
                
                if st.button("Check Again", use_container_width=True, type="primary"):
                    st.session_state.poll_start_time = time.time()
                    st.rerun()
                
                if st.button("Go to Chat Anyway", use_container_width=True):
                    st.warning("Insights may not be available yet in chat.")
                    st.switch_page("pages/chatbot.py")

if __name__ == "__main__":
    main()
