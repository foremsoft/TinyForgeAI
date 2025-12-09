"""
TinyForgeAI Playground - Streamlit UI for testing inference services.

Usage:
    cd playground
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import requests
import json
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="TinyForgeAI Playground",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 4px solid #c62828;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔧 TinyForgeAI Playground")
st.markdown("Test your TinyForgeAI inference services interactively.")

# Sidebar configuration
st.sidebar.header("Configuration")

api_url = st.sidebar.text_input(
    "Service URL",
    value="http://127.0.0.1:8000/predict",
    help="URL of your TinyForgeAI inference service"
)

timeout = st.sidebar.slider(
    "Request Timeout (seconds)",
    min_value=1,
    max_value=30,
    value=5
)

# Health check button
if st.sidebar.button("Check Health"):
    health_url = api_url.rsplit("/", 1)[0] + "/health"
    try:
        resp = requests.get(health_url, timeout=timeout)
        if resp.status_code == 200:
            st.sidebar.success(f"✅ Service is healthy")
            st.sidebar.json(resp.json())
        else:
            st.sidebar.error(f"❌ Health check failed: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.sidebar.error("❌ Cannot connect to service")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Quick Start:**
1. Start your inference service:
   ```
   foremforge serve --dir ./service --port 8000
   ```
2. Enter your input text
3. Click "Send Request"
""")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")

    input_text = st.text_area(
        "Enter your text",
        value="What is your refund policy?",
        height=150,
        help="Enter the text you want to send to the model"
    )

    # Additional options
    with st.expander("Advanced Options"):
        include_metadata = st.checkbox("Include metadata in request", value=False)

        if include_metadata:
            metadata_str = st.text_area(
                "Metadata (JSON)",
                value='{"source": "playground"}',
                height=100
            )

    send_button = st.button("🚀 Send Request", type="primary", use_container_width=True)

with col2:
    st.subheader("Response")

    if send_button:
        if not input_text.strip():
            st.error("Please enter some input text")
        else:
            # Prepare request
            payload = {"input": input_text}

            if include_metadata:
                try:
                    payload["metadata"] = json.loads(metadata_str)
                except json.JSONDecodeError:
                    st.error("Invalid JSON in metadata field")
                    payload = None

            if payload:
                with st.spinner("Sending request..."):
                    try:
                        response = requests.post(
                            api_url,
                            json=payload,
                            timeout=timeout,
                            headers={"Content-Type": "application/json"}
                        )

                        if response.status_code == 200:
                            result = response.json()
                            st.success("Request successful!")

                            # Display response
                            if "output" in result:
                                st.markdown("**Output:**")
                                st.info(result["output"])

                            if "confidence" in result:
                                confidence = result["confidence"]
                                st.markdown(f"**Confidence:** {confidence:.2%}")
                                st.progress(confidence)

                            # Full JSON response
                            with st.expander("View Full Response"):
                                st.json(result)
                        else:
                            st.error(f"Request failed with status {response.status_code}")
                            st.code(response.text)

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to the service. Make sure it's running.")
                        st.markdown("""
                        **To start the service:**
                        ```bash
                        foremforge serve --dir ./your_service --port 8000
                        ```
                        """)
                    except requests.exceptions.Timeout:
                        st.error(f"❌ Request timed out after {timeout} seconds")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# Example section
st.markdown("---")
st.subheader("📝 Example Inputs")

example_inputs = [
    "Hello, how are you?",
    "What is your return policy?",
    "Tell me about your product features",
    "How do I contact customer support?",
]

cols = st.columns(len(example_inputs))
for i, example in enumerate(example_inputs):
    with cols[i]:
        if st.button(example[:20] + "...", key=f"example_{i}"):
            st.session_state["input_text"] = example
            st.rerun()

# cURL equivalent
st.markdown("---")
st.subheader("🔧 cURL Equivalent")

curl_command = f'''curl -X POST {api_url} \\
  -H "Content-Type: application/json" \\
  -d '{{"input": "{input_text}"}}'
'''

st.code(curl_command, language="bash")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>TinyForgeAI Playground |
    <a href="https://github.com/anthropics/TinyForgeAI">GitHub</a> |
    <a href="https://github.com/anthropics/TinyForgeAI/tree/main/docs">Documentation</a>
    </p>
</div>
""", unsafe_allow_html=True)
