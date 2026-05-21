import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import re
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Dark Trace AI - Message Analysis",
    page_icon="🔍",
    layout="wide"
)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

def get_feature_details(features):
    """Get detailed feature analysis for a message from API features"""
    details = {
        "Text Analysis": {
            "Character Count": features.get('char_count', 0),
            "Word Count": features.get('word_count', 0),
            "Total Emojis": features.get('emoji_count', 0),
            "Drug-Related Emojis": features.get('drug_emoji_count', 0),
            "Slang Words": features.get('slang_count', 0)
        },
        "Pattern Detection": {
            "Price Patterns": "Yes" if features.get('has_price') else "No",
            "Quantity Patterns": "Yes" if features.get('has_quantity') else "No",
            "Location Indicators": "Yes" if features.get('has_location') else "No",
            "Urgency Indicators": "Yes" if features.get('has_urgency') else "No"
        },
        "Text Characteristics": {
            "Has Punctuation": "Yes" if features.get('has_punctuation') else "No",
            "Has Capitalization": "Yes" if features.get('has_capitalization') else "No"
        }
    }
    
    return details


# App title and description
st.title("🔍 Dark Trace AI - Message Analysis")
st.markdown("""
This tool analyzes messages for potential suspicious content using advanced AI.
Enter a message below to get real-time analysis and risk assessment.
""")

# Sidebar with options
st.sidebar.title("Analysis Options")
threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Adjust the sensitivity of the risk detection"
)

show_details = st.sidebar.checkbox(
    "Show Detailed Analysis",
    value=True,
    help="Display detailed feature analysis"
)

# Add reset button
if st.sidebar.button("🔄 Reset Analysis", help="Clear current analysis and start fresh"):
    st.rerun()

# --- Model Version Display ---
def get_model_version_info():
    try:
        import requests
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'FastAPI Mock v1.0'
    except Exception:
        pass
    return None, None

mod_time, model_version = get_model_version_info()
st.sidebar.markdown('---')
st.sidebar.markdown('**Model Info:**')
if model_version:
    st.sidebar.markdown(f"- Version: `{model_version}`")
if mod_time:
    st.sidebar.markdown(f"- Last updated: `{mod_time}`")
# --- End Model Version Display ---

# Initialize session state for message history
if 'message_history' not in st.session_state:
    st.session_state.message_history = []

# Input area
message = st.text_area(
    "Enter message to analyze:",
    height=100,
    placeholder="Type or paste message here...",
    key="message_input"
)

if message:
    try:
        import requests
        # Send HTTP POST to FastAPI backend
        response = requests.post(f"{API_URL}/predict", json={"message": message})
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            features = result["features"]
            
            # Store in session state
            analysis_result = {
                'message': message,
                'prediction': prediction,
                'features': features,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            st.session_state.message_history.append(analysis_result)
            
            # Display result
            st.markdown("### Analysis Results")
            
            # Risk meter
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.progress(prediction)
                risk_level = "High" if prediction >= 0.7 else "Medium" if prediction >= 0.4 else "Low"
                st.markdown(f"**Risk Score:** {prediction:.2%} ({risk_level} Risk)")
            
            if show_details:
                st.markdown("### Detailed Analysis")
                details = get_feature_details(features)
                
                # Display feature details in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Text Analysis")
                    st.json(details["Text Analysis"])
                    st.markdown("#### 🔍 Pattern Detection")
                    st.json(details["Pattern Detection"])
                    
                with col2:
                    st.markdown("#### 📝 Text Characteristics")
                    st.json(details["Text Characteristics"])
                    
                    # Show risk factors
                    st.markdown("#### ⚠️ Risk Factors")
                    risk_factors = []
                    if details["Text Analysis"]["Drug-Related Emojis"] > 0:
                        risk_factors.append("🚨 Drug-related emojis detected")
                    if details["Pattern Detection"]["Price Patterns"] == "Yes":
                        risk_factors.append("💰 Price patterns detected")
                    if details["Pattern Detection"]["Quantity Patterns"] == "Yes":
                        risk_factors.append("📏 Quantity patterns detected")
                    if details["Pattern Detection"]["Location Indicators"] == "Yes":
                        risk_factors.append("📍 Location indicators detected")
                    if details["Pattern Detection"]["Urgency Indicators"] == "Yes":
                        risk_factors.append("⏰ Urgency indicators detected")
                    if details["Text Analysis"]["Slang Words"] > 0:
                        risk_factors.append("🗣️ Slang words detected")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.markdown(f"- {factor}")
                    else:
                        st.markdown("✅ No obvious risk factors detected")
            
            # Warning for high-risk messages
            if prediction >= 0.7:
                st.error("🚨 **HIGH RISK:** This message shows strong indicators of suspicious content.")
            elif prediction >= 0.4:
                st.warning("⚠️ **MEDIUM RISK:** This message shows some indicators of suspicious content.")
            else:
                st.success("✅ **LOW RISK:** This message appears to be normal.")
                
            # Clear the input for next message
            st.markdown("---")
            st.markdown("💡 **Tip:** Enter a new message above to analyze another text.")
            
        else:
            st.error(f"Error from API (Status {response.status_code}): {response.text}")
            
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        st.info("Please ensure the FastAPI backend is running.")

# Display message history
if st.session_state.message_history:
    st.markdown("---")
    st.markdown("### 📝 Analysis History")
    
    for i, result in enumerate(st.session_state.message_history[-5:], 1):  # Show last 5
        with st.expander(f"Message {i} ({result['timestamp']}) - {result['prediction']:.1%} risk"):
            st.write(f"**Message:** {result['message']}")
            st.write(f"**Risk Score:** {result['prediction']:.2%}")
            
            # Show key features
            features = result['features']
            st.write("**Key Features:**")
            if features.get('drug_emoji_count', 0) > 0:
                st.write(f"- 🚨 Drug emojis: {features['drug_emoji_count']}")
            if features.get('slang_count', 0) > 0:
                st.write(f"- 🗣️ Slang words: {features['slang_count']}")
            if features.get('has_price'):
                st.write("- 💰 Price patterns detected")
            if features.get('has_quantity'):
                st.write("- 📏 Quantity patterns detected")
            if features.get('has_location'):
                st.write("- 📍 Location indicators detected")
            
            # Add delete button for each history item
            if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                st.session_state.message_history.pop(i-1)
                st.rerun()

# Footer
st.markdown("---")
st.markdown("*Note: This tool is for demonstration purposes only. Always verify results manually.*")
