import streamlit as st
import streamlit.components.v1 as components
import boto3
from utils.session_manager import SessionManager
from utils.api_client import APIClient

# Initialize S3 client
s3_client = boto3.client('s3')
sts_client = boto3.client("sts")
ACCOUNT_ID = sts_client.get_caller_identity()['Account']
S3_BUCKET_INSIGHTS = 'generated-insights'
S3_BUCKET_INSIGHTS = f"{S3_BUCKET_INSIGHTS}-{ACCOUNT_ID}"
S3_QUESTIONNAIRES = 'icarus-questionnaires'
S3_QUESTIONNAIRES = f"{S3_QUESTIONNAIRES}-{ACCOUNT_ID}"

st.set_page_config(
    page_title="Project Icarus - Chat",
    page_icon="💬",
    layout="wide"
)

# Layout, spacing, and login-top-right CSS (unchanged)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1rem !important;
        margin-top: -35px !important;
    }
    h1:first-of-type {
        padding-top: 0.5rem !important;
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
    }
    h2, h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .element-container {
        margin-bottom: 0rem !important;
    }
    iframe {
        margin-bottom: 0rem !important;
        margin-top: 0rem !important;
    }
    .stButton > button {
        font-weight: 600;
    }
    .send-btn > button {
        background-color: #31a354 !important;
        color: white;
    }
    .clear-btn > button {
        background-color: #e74c3c !important;
        color: white;
    }
    .stTextArea {
        margin-top: 0rem !important;
        margin-bottom: 0.1rem !important;
    }
    .stDownloadButton {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    hr {
        margin: 0.25rem 0 !important;
    }
    [data-testid="column"] > div {
        padding-top: 0 !important;
    }
    .stAlert {
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
    }
    .login-right {
        position: absolute;
        top: -40px;
        right: 32px;
        z-index: 10;
        font-size: 0.9rem;
        color: #999;
    }
    .login-right a { color: #69c; text-decoration: none;}
</style>
""", unsafe_allow_html=True)

SessionManager.initialize_session()

if not SessionManager.is_authenticated():
    st.error("Please sign in first")
    if st.button("Sign in/Sign up", use_container_width=True):
        st.switch_page("pages/auth.py")
    st.stop()

if not SessionManager.check_questionnaire_exists():
    st.warning("Please complete the questionnaire first")
    if st.button("Fill questionnaire", use_container_width=True):
        st.switch_page('pages/questionnaire.py')
    st.stop()

def load_insights_from_s3(email):
    if 'insights_content' in st.session_state and st.session_state.insights_content:
        return st.session_state.insights_content
    username = email.split("@")[0]
    try:
        insights_key = f"{username}/{username}_insights.md"
        response = s3_client.get_object(
            Bucket=S3_BUCKET_INSIGHTS,
            Key=insights_key
        )
        content = response['Body'].read().decode('utf-8')
        st.session_state.insights_content = content
        return content
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        st.error(f"Error loading insights: {e}")
        return None

def create_scrollable_html(content, height_px=500, bg_color="#ffffff"):
    import json
    safe_content = json.dumps(content) if content else '""'
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }}
            #content {{
                height: {height_px}px;
                overflow-y: auto;
                overflow-x: hidden;
                padding: 20px;
                background-color: {bg_color};
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                line-height: 1.6;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <div id="content"></div>
        <script>
            const markdownContent = {safe_content};
            document.getElementById('content').innerHTML = marked.parse(markdownContent);
        </script>
    </body>
    </html>
    """
    return html_code

def create_chat_html(messages, height_px=500, bg_color="#f9f9f9"):
    import json
    messages_json = json.dumps([{
        "role": msg["role"],
        "content": msg["content"][0]["text"]
    } for msg in messages])
    chat_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
            #chat-container {{
                height: {height_px}px; overflow-y: auto; overflow-x: hidden; padding: 15px;
                background-color: {bg_color}; border: 1px solid #e0e0e0; border-radius: 8px;
            }}
            .message {{ margin-bottom: 15px; padding: 12px; border-radius: 8px; line-height: 1.5; }}
            .user-message {{ background-color: #e3f2fd; border-left: 4px solid #2196f3; }}
            .assistant-message {{ background-color: #f5f5f5; border-left: 4px solid #4caf50; }}
            .message-header {{ font-weight: bold; margin-bottom: 5px; }}
            .user-header {{ color: #1976d2; }}
            .assistant-header {{ color: #388e3c; }}
            .message-content {{ color: #333; }}
        </style>
    </head>
    <body>
        <div id="chat-container"></div>
        <script>
            const messages = {messages_json};
            const container = document.getElementById('chat-container');
            messages.forEach(message => {{
                const messageDiv = document.createElement('div');
                messageDiv.className = message.role === 'user' ? 'message user-message' : 'message assistant-message';
                const headerDiv = document.createElement('div');
                headerDiv.className = message.role === 'user' ? 'message-header user-header' : 'message-header assistant-header';
                headerDiv.textContent = message.role === 'user' ? '👤 You:' : '🤖 Assistant:';
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.innerHTML = marked.parse(message.content);
                messageDiv.appendChild(headerDiv);
                messageDiv.appendChild(contentDiv);
                container.appendChild(messageDiv);
            }});
            container.scrollTop = container.scrollHeight;
        </script>
    </body>
    </html>
    """
    return chat_html

def main():
    st.title("🎭 Campaign Strategist")
    st.markdown(
    f"""
    <div style="position:absolute;top:-65px;right:32px;z-index:10;">
        <span style="font-size:0.9rem;color:#999;">
            Logged in as: <a href="mailto:{SessionManager.get_user()}" style="color:#69c;text-decoration:none;">{SessionManager.get_user()}</a>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

    # Load insights
    email = SessionManager.get_user()
    insights_content = load_insights_from_s3(email)

    # Handle clear_input flag for text area
    if st.session_state.get("clear_input"):
        st.session_state["user_input"] = ""
        st.session_state["clear_input"] = False

    # Columns: chat, insights
    col_chat, col_insights = st.columns([1, 2])

    # Insights (right column)
    with col_insights:
        st.subheader("📊 Campaign Insights")
        if insights_content:
            insights_html = create_scrollable_html(insights_content, height_px=500, bg_color="#fffef9")
            components.html(insights_html, height=520, scrolling=False)
            st.download_button(
                label="📥 Download Insights",
                data=insights_content,
                file_name=f"{email}_campaign_insights.md",
                mime="text/markdown",
                use_container_width=True
            )
        else:
            st.warning("⏳ Your campaign insights are still being generated.")
            st.info("""
**What's happening:**
- Historical election data is being analyzed
- Win Gap scenarios are being calculated
- Precinct-level strategies are being formulated

This process typically takes 1-2 minutes.
            """)
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
            if st.button("← Back to Questionnaire", use_container_width=True):
                st.switch_page('pages/questionnaire.py')
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            SessionManager.logout()
            st.session_state.current_page = "auth"
            st.rerun()

    # Chat (left column)
    with col_chat:
        st.subheader("💬 Chat")
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        if st.session_state.chat_messages:
            chat_html = create_chat_html(st.session_state.chat_messages, height_px=500)
            components.html(chat_html, height=520, scrolling=False)
        else:
            st.info("No messages yet. Start a conversation below!")

        # Input area, tightly below the chat iframe
        # -- Text input
        user_input = st.text_area(
            "",
            placeholder="e.g., How should I focus my door-knocking efforts?",
            height=80,
            key="user_input"
        )

        # -- Buttons in 2 columns (green send, red clear)
        send_col, clear_col = st.columns([3, 1])
        with send_col:
            send_clicked = st.button("Send Message", key="sendmsg", use_container_width=True, type='primary')
        with clear_col:
            clear_clicked = st.button("Clear Chat", key="clearchat", use_container_width=True, type='secondary')

        # Custom button styling - attach class for color
        st.markdown(
            """
            <style>
            div[data-testid="column"] .stButton button[kind="secondary"]:nth-of-type(1) {
                background-color: #31a354 !important;
                color: white !important;
            }
            div[data-testid="column"] .stButton button[kind="secondary"]:nth-of-type(2) {
                background-color: #e74c3c !important;
                color: white !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Handle "Send Message"
        if send_clicked:
            if not user_input.strip():
                st.error("Please enter a message")
            else:
                chat_history = SessionManager.get_chat_history()
                with st.spinner("Thinking..."):
                    api_client = APIClient()
                    response = api_client.send_chat_message(
                        SessionManager.get_user(),
                        user_input,
                        chat_history
                    )
                    if response.get("success"):
                        SessionManager.add_chat_message(role="user", content=user_input)
                        SessionManager.add_chat_message(
                            role="assistant",
                            content=response.get("message", "No response")
                        )
                        st.session_state["clear_input"] = True
                        st.rerun()
                    else:
                        st.error(f"Error: {response.get('message', 'Unknown error')}")

        # Handle "Clear Chat"
        if clear_clicked:
            st.session_state.chat_messages = []
            st.session_state["clear_input"] = True
            st.rerun()

if __name__ == "__main__":
    main()
