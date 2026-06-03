import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import re
import os
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

# Custom transformer classes needed for model loading
class LexWrap(BaseEstimator, TransformerMixin):
    """Wrapper class for lexicon features - required for model loading"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Handle both 'lex_dummy' and 'enhanced_features' (or any single-column DataFrame)
        if isinstance(X, pd.DataFrame):
            # If a single column is provided via ColumnTransformer, stack that column
            if X.shape[1] == 1:
                col = X.columns[0]
                return np.stack(X[col].values)
            # If multiple columns, try known names and otherwise fall back to first
            for candidate in ('lex_dummy', 'enhanced_features'):
                if candidate in X.columns:
                    return np.stack(X[candidate].values)
            return np.stack(X.iloc[:, 0].values)
        # Fallback for array-like inputs
        try:
            return np.stack(X)
        except Exception:
            return np.asarray(X)

class LexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    """Lexicon-based feature extractor - required for model loading"""
    def __init__(self, text_col='message_text'):
        self.text_col = text_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Basic implementation - this may need adjustment based on the actual model
        if isinstance(X, pd.Series):
            texts = X.fillna('').astype(str)
        else:
            texts = X[self.text_col].fillna('').astype(str)
            
        # Simple feature extraction
        features = np.zeros((len(texts), 6))
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            # Basic lexicon features
            features[i, 0] = len(re.findall(r'[\U0001F300-\U0001F9FF]', text))  # emoji count
            features[i, 1] = len(re.findall(r'\b(?:weed|pot|grass|mary|jane)\b', text_lower))  # slang count
            features[i, 2] = len(re.findall(r'(?:\$|Γé╣)\s?\d+', text))  # price patterns
            features[i, 3] = int(bool(re.search(r'meet|spot|drop|location|pin|dm|pm', text_lower)))  # location
            features[i, 4] = int(bool(re.search(r'got|fresh batch|in stock|moving fast', text_lower)))  # availability
            features[i, 5] = int(bool(re.search(r'crypto|btc|upi|gpay|paytm|card', text_lower)))  # payment
            
        return features

# Set page config
st.set_page_config(
    page_title="Dark Trace AI - Message Analysis",
    page_icon="≡ƒöì",
    layout="wide"
)

# Load model and resources
MODEL_PATH = Path('artifacts/improved_pipeline.joblib')
EMOJI_DICT_PATH = Path('artifacts/emoji_dict.csv')
SLANG_DICT_PATH = Path('artifacts/slang_dict.csv')

@st.cache_resource
def load_resources():
    """Load model and dictionaries with proper error handling"""
    try:
        if not MODEL_PATH.exists():
            st.error(f"Model file not found at {MODEL_PATH}")
            st.stop()
            
        if not EMOJI_DICT_PATH.exists():
            st.error(f"Emoji dictionary not found at {EMOJI_DICT_PATH}")
            st.stop()
            
        if not SLANG_DICT_PATH.exists():
            st.error(f"Slang dictionary not found at {SLANG_DICT_PATH}")
            st.stop()
            
        model = joblib.load(MODEL_PATH)
        emoji_dict = pd.read_csv(EMOJI_DICT_PATH)
        slang_dict = pd.read_csv(SLANG_DICT_PATH)
        
        # Validate loaded data
        if emoji_dict.empty:
            st.warning("Emoji dictionary is empty")
        if slang_dict.empty:
            st.warning("Slang dictionary is empty")
            
        return model, emoji_dict, slang_dict
        
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        st.error("Please ensure all required files are present in the artifacts directory.")
        st.stop()

def create_input_dataframe(message, platform='unknown', message_type='text'):
    """
    Create a DataFrame with the exact structure expected by the trained model.
    The model expects columns: message_text, platform, message_type, lex_dummy
    """
    # Create a simple lex_dummy feature based on the message content
    # This is a placeholder - in the real training, this would be the enhanced features
    lex_dummy = create_lex_dummy_feature(message)
    
    return pd.DataFrame({
        'message_text': [message],
        'platform': [platform],
        'message_type': [message_type],
        'lex_dummy': [lex_dummy]
    })

def create_lex_dummy_feature(message):
    """
    Create lex_dummy feature using the same LexiconFeatureExtractor as in training.
    This should return exactly 6 features to match the training pipeline.
    """
    # Use the same LexiconFeatureExtractor as in training
    lex_extractor = LexiconFeatureExtractor('message_text')
    
    # Create a DataFrame with the message
    df = pd.DataFrame({'message_text': [message]})
    
    # Transform to get the 6 features
    features = lex_extractor.transform(df)
    
    # Return the first row (since we only have one message)
    return features[0]

def get_feature_details(text):
    """Get detailed feature analysis for a message"""
    lex_features = create_lex_dummy_feature(text)
    
    details = {
        "Lexicon Analysis": {
            "Slang Word Count": int(lex_features[0]),
            "Emoji Count": int(lex_features[1]),
            "Price Patterns": int(lex_features[2]),
            "Location Indicators": "Yes" if lex_features[3] > 0 else "No",
            "Availability Terms": "Yes" if lex_features[4] > 0 else "No",
            "Payment Methods": "Yes" if lex_features[5] > 0 else "No"
        }
    }
    
    return details

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1e3c72;
    }
    .risk-high { border-left-color: #ff4444; }
    .risk-medium { border-left-color: #ffaa44; }
    .risk-low { border-left-color: #44ff44; }
    .feature-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App title and description with enhanced styling
st.markdown("""
<div class="main-header">
    <h1>≡ƒöì Dark Trace AI - Message Analysis</h1>
    <p>Advanced AI-powered detection of suspicious communication patterns</p>
    <p><em>Real-time analysis ΓÇó Behavioral insights ΓÇó Risk assessment</em></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar with enhanced options
st.sidebar.title("ΓÜÖ∩╕Å Analysis Options")

# Analysis settings
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

# Example messages section
st.sidebar.markdown("---")
st.sidebar.markdown("### ≡ƒô¥ Example Messages")
st.sidebar.markdown("Click to try these examples:")

example_messages = {
    "Normal Message": "Hey, how are you doing today? Want to grab coffee later?",
    "Suspicious (High)": "Got that good stuff ≡ƒìü≡ƒÆè $50 quick meet behind mall ≡ƒÆ░ DM me ASAP",
    "Suspicious (Medium)": "You still looking? I got what you need. Hit me up when ready.",
    "Business": "Meeting scheduled for 3 PM. Please bring the quarterly reports."
}

selected_example = st.sidebar.selectbox(
    "Choose an example:",
    [""] + list(example_messages.keys()),
    help="Select an example message to analyze"
)

# Information section
st.sidebar.markdown("---")
st.sidebar.markdown("### Γä╣∩╕Å About This Tool")
st.sidebar.markdown("""
This AI system analyzes:
- **Emoji patterns** and combinations
- **Slang terminology** usage
- **Behavioral indicators** in text
- **Context clues** and urgency
- **Communication style** patterns

**Disclaimer:** This is a demonstration tool. 
Results should be verified manually.
""")

# Statistics section
st.sidebar.markdown("---")
st.sidebar.markdown("### ≡ƒôè Quick Stats")
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
    
st.sidebar.metric("Analyses Performed", st.session_state.analysis_count)

# --- Model Version Display ---


def get_model_version_info():
    try:
        stat = os.stat(MODEL_PATH)
        mod_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        model, _, _ = load_resources()
        version = getattr(model, 'version', None)
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

# Input validation function
def validate_input(text):
    """Validate and sanitize user input"""
    if not text or not text.strip():
        return False, "Please enter a message to analyze."
    
    # Check for reasonable length limits
    if len(text) > 10000:
        return False, "Message is too long. Please limit to 10,000 characters."
    
    # Basic security check - prevent potential injection attempts
    suspicious_patterns = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'eval\s*\(',
        r'exec\s*\('
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, text.lower()):
            return False, "Input contains potentially unsafe content."
    
    return True, ""

# Handle example message selection
default_message = ""
if selected_example and selected_example in example_messages:
    default_message = example_messages[selected_example]

# Input area
st.markdown("### ≡ƒô¥ Message Input")
message = st.text_area(
    "Enter message to analyze:",
    value=default_message,
    height=120,
    placeholder="Type or paste message here, or select an example from the sidebar...",
    max_chars=10000,
    help="Enter the message you want to analyze for suspicious content. Maximum 10,000 characters."
)

# Clear button
col1, col2, col3 = st.columns([1,1,4])
with col1:
    if st.button("≡ƒùæ∩╕Å Clear", help="Clear the input field"):
        st.rerun()
with col2:
    if st.button("≡ƒôï Analyze", help="Analyze the current message", type="primary"):
        if message:
            st.session_state.force_analysis = True

# Only load resources if running in Streamlit context
try:
    model, _, _ = load_resources()
except Exception:
    # This will happen when importing the module outside of Streamlit
    model, _, _ = None, None, None

if message:
    # Validate input first
    is_valid, error_msg = validate_input(message)
    
    if not is_valid:
        st.error(error_msg)
    elif model is None:
        st.error("Model not loaded. Please ensure all required files are present in the artifacts directory.")
    else:
        # Increment analysis counter
        st.session_state.analysis_count += 1
        
        try:
            # Create input DataFrame with the exact structure expected by the model
            input_df = create_input_dataframe(message)
            
            # Get prediction using the model pipeline
            prediction_proba = model.predict_proba(input_df)
            
            # Handle different model outputs
            if prediction_proba.shape[1] == 2:
                prediction = prediction_proba[0, 1]  # Binary classification
            else:
                prediction = prediction_proba[0, 0]  # Single output
            
            # Ensure prediction is within valid range
            prediction = max(0.0, min(1.0, prediction))
            
            # Display result
            st.markdown("### ≡ƒôè Analysis Results")
            
            # Risk meter with better visual design
            col1, col2, col3 = st.columns([1,3,1])
            with col2:
                # Create a more informative progress bar
                risk_color = "red" if prediction >= 0.7 else "orange" if prediction >= 0.4 else "green"
                st.progress(prediction)
                
                risk_level = "≡ƒö┤ High" if prediction >= 0.7 else "≡ƒƒí Medium" if prediction >= 0.4 else "≡ƒƒó Low"
                st.markdown(f"**Risk Score:** {prediction:.1%} ({risk_level} Risk)")
                
                # Add confidence indicator
                confidence = "High" if prediction >= 0.8 or prediction <= 0.2 else "Medium"
                st.caption(f"Confidence: {confidence}")
            
            if show_details:
                st.markdown("### ≡ƒöì Detailed Analysis")
                try:
                    details = get_feature_details(message)
                    
                    # Display feature details in columns with better formatting
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### ≡ƒöì Lexicon Analysis")
                        with st.expander("View Details", expanded=True):
                            st.json(details["Lexicon Analysis"])
                        
                    with col2:
                        st.markdown("#### ≡ƒôè Feature Summary")
                        st.info("""
                        **Analysis includes:**
                        - Slang terminology detection
                        - Emoji pattern analysis  
                        - Price/quantity indicators
                        - Location references
                        - Availability signals
                        - Payment method mentions
                        """)
                            
                except Exception as e:
                    st.warning(f"Could not generate detailed analysis: {str(e)}")
            
            # Enhanced warning system
            st.markdown("### ΓÜá∩╕Å Risk Assessment")
            if prediction >= 0.7:
                st.error("≡ƒÜ¿ **HIGH RISK**: This message shows strong indicators of suspicious content.")
                st.markdown("**Recommended Actions:**")
                st.markdown("- Review message context and sender")
                st.markdown("- Consider additional verification")
                st.markdown("- Flag for manual review")
            elif prediction >= 0.4:
                st.warning("ΓÜá∩╕Å **MEDIUM RISK**: This message shows some indicators of suspicious content.")
                st.markdown("**Recommended Actions:**")
                st.markdown("- Monitor conversation patterns")
                st.markdown("- Consider context and sender history")
            else:
                st.success("Γ£à **LOW RISK**: This message appears to be normal communication.")
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.error("Please try again or contact support if the problem persists.")
            
            # Log error details for debugging (in a real app, this would go to logs)
            with st.expander("Error Details (for debugging)"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())

# Enhanced Footer
st.markdown("---")
st.markdown("### ≡ƒôÜ Additional Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**≡ƒöÆ Privacy & Security**")
    st.markdown("""
    - Messages are processed locally
    - No data is stored permanently  
    - Input validation prevents injection
    - Secure model loading
    """)

with col2:
    st.markdown("**≡ƒÄ» Accuracy Notes**")
    st.markdown("""
    - AI predictions are probabilistic
    - Context matters significantly
    - False positives/negatives possible
    - Manual verification recommended
    """)

with col3:
    st.markdown("**≡ƒ¢á∩╕Å Technical Details**")
    st.markdown("""
    - Machine Learning pipeline
    - Multi-feature analysis
    - Real-time processing
    - Behavioral pattern detection
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p><strong>Dark Trace AI - Message Analysis Tool</strong></p>
    <p><em>For research and demonstration purposes only. Always verify results manually.</em></p>
    <p>Built with Streamlit ΓÇó Powered by scikit-learn ΓÇó Enhanced with custom NLP features</p>
</div>
""", unsafe_allow_html=True)
