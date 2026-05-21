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

# Load model and resources
MODEL_PATH = Path('artifacts/rule_based_classifier.joblib')  # Use the rule-based classifier
EMOJI_DICT_PATH = Path('artifacts/emoji_dict.csv')
SLANG_DICT_PATH = Path('artifacts/slang_dict.csv')

@st.cache_resource
def load_resources():
    try:
        # Load dictionaries first
        emoji_dict = pd.read_csv(EMOJI_DICT_PATH)
        slang_dict = pd.read_csv(SLANG_DICT_PATH)
        
        # Now load the rule-based classifier
        classifier = joblib.load(MODEL_PATH)
        
        return classifier, emoji_dict, slang_dict
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        raise e

def extract_features_for_display(text, emoji_dict, slang_dict):
    """Extract features for display purposes"""
    text_lower = text.lower()
    words = text_lower.split()
    
    features = {}
    
    # Emoji count
    emojis = re.findall(r'[\U0001F300-\U0001F9FF]', text)
    features['emoji_count'] = len(emojis)
    
    # Drug-related emojis
    drug_emojis = {'🍁', '💊', '💉', '🔥', '💨', '🌿', '🌱', '🪴', '🍄', '🔌'}
    drug_emoji_count = sum(1 for e in emojis if e in drug_emojis)
    features['drug_emoji_count'] = drug_emoji_count
    features['has_drug_emoji'] = drug_emoji_count > 0
    
    # Slang count
    slang_words = set(w.lower() for w in slang_dict['slang_term'])
    text_slang = [w for w in words if w in slang_words]
    features['slang_count'] = len(text_slang)
    features['has_slang'] = len(text_slang) > 0
    
    # Pattern detection
    features['has_price'] = bool(re.search(r'\$\d+|\d+\s*dollars?|\d+\s*bucks?', text_lower))
    features['has_quantity'] = bool(re.search(r'\d+\s*(?:g|gram|oz|ounce|lb|pound|k|kilo)', text_lower))
    features['has_location'] = bool(re.search(r'spot|location|place|meet|behind|alley|corner|parking', text_lower))
    features['has_urgency'] = bool(re.search(r'asap|quick|fast|rush|hurry|now|urgent|immediate', text_lower))
    
    # Text characteristics
    features['char_count'] = len(text)
    features['word_count'] = len(words)
    features['has_punctuation'] = bool(re.search(r'[.!?]', text))
    features['has_capitalization'] = bool(re.search(r'[A-Z]', text))
    
    return features

def get_feature_details(text, emoji_dict, slang_dict):
    """Get detailed feature analysis for a message"""
    features = extract_features_for_display(text, emoji_dict, slang_dict)
    
    details = {
        "Text Analysis": {
            "Character Count": features['char_count'],
            "Word Count": features['word_count'],
            "Total Emojis": features['emoji_count'],
            "Drug-Related Emojis": features['drug_emoji_count'],
            "Slang Words": features['slang_count']
        },
        "Pattern Detection": {
            "Price Patterns": "Yes" if features['has_price'] else "No",
            "Quantity Patterns": "Yes" if features['has_quantity'] else "No",
            "Location Indicators": "Yes" if features['has_location'] else "No",
            "Urgency Indicators": "Yes" if features['has_urgency'] else "No"
        },
        "Text Characteristics": {
            "Has Punctuation": "Yes" if features['has_punctuation'] else "No",
            "Has Capitalization": "Yes" if features['has_capitalization'] else "No"
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
        stat = os.stat(MODEL_PATH)
        mod_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        classifier, _, _ = load_resources()
        version = getattr(classifier, 'version', 'Rule-Based v1.0')
        return mod_time, version
    except Exception:
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

# Initialize and load classifier
try:
    classifier, emoji_dict, slang_dict = load_resources()
    
    if message:
        try:
            # Get prediction using the rule-based classifier
            prediction = classifier.predict_proba([message])[0, 1]
            
            # Store in session state
            analysis_result = {
                'message': message,
                'prediction': prediction,
                'features': extract_features_for_display(message, emoji_dict, slang_dict),
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
                details = get_feature_details(message, emoji_dict, slang_dict)
                
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
                
        except Exception as e:
            st.error(f"Error analyzing message: {str(e)}")
            st.info("Please check if the classifier files are properly loaded and try again.")
            
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
                if features['drug_emoji_count'] > 0:
                    st.write(f"- 🚨 Drug emojis: {features['drug_emoji_count']}")
                if features['slang_count'] > 0:
                    st.write(f"- 🗣️ Slang words: {features['slang_count']}")
                if features['has_price']:
                    st.write("- 💰 Price patterns detected")
                if features['has_quantity']:
                    st.write("- 📏 Quantity patterns detected")
                if features['has_location']:
                    st.write("- 📍 Location indicators detected")
                
                # Add delete button for each history item
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    st.session_state.message_history.pop(i-1)
                    st.rerun()
            
except Exception as e:
    st.error(f"Error initializing the application: {str(e)}")
    st.info("Please ensure all required files are present in the artifacts directory.")

# Footer
st.markdown("---")
st.markdown("*Note: This tool is for demonstration purposes only. Always verify results manually.*")
