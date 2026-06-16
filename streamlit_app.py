import streamlit as st
import pandas as pd
import os
import re
import requests
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Dark Trace AI - Message Analysis",
    page_icon="🔍",
    layout="wide"
)

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
    <h1>🔍 Dark Trace AI - Message Analysis</h1>
    <p>Advanced AI-powered detection of suspicious communication patterns</p>
    <p><em>Real-time analysis • Behavioral insights • Risk assessment</em></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# API configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Sidebar with enhanced options
st.sidebar.title("⚙️ Analysis Options")

show_details = st.sidebar.checkbox(
    "Show Detailed Analysis",
    value=True,
    help="Display detailed feature analysis"
)

# Example messages section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Example Messages")
st.sidebar.markdown("Click to try these examples:")

example_messages = {
    "Normal Message": "Hey, how are you doing today? Want to grab coffee later?",
    "Suspicious (High)": "Got that good stuff 🍁💊 $50 quick meet behind mall 💰 DM me ASAP",
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
st.sidebar.markdown("### ℹ️ About This Tool")
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
st.sidebar.markdown("### 📊 Quick Stats")
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
    
st.sidebar.metric("Analyses Performed", st.session_state.analysis_count)

# --- Model Version Display ---
def get_model_version_info():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'FastAPI Model v1.0'
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

# Input validation function
def validate_input(text):
    """Validate and sanitize user input"""
    if not text or not text.strip():
        return False, "Please enter a message to analyze."
    
    if len(text) > 10000:
        return False, "Message is too long. Please limit to 10,000 characters."
    
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

# Handle example message selection
default_message = ""
if selected_example and selected_example in example_messages:
    default_message = example_messages[selected_example]

# Input area
st.markdown("### 📝 Message Input")
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
    if st.button("🗑️ Clear", help="Clear the input field"):
        st.rerun()
with col2:
    analyze_button = st.button("🔍 Analyze", help="Analyze the current message", type="primary")

if analyze_button and message:
    is_valid, error_msg = validate_input(message)
    
    if not is_valid:
        st.error(error_msg)
    else:
        st.session_state.analysis_count += 1
        try:
            # Add explain=true to request SHAP values
            response = requests.post(f"{API_URL}/predict?explain=true", json={"message": message})
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                features = result["features"]
                
                prediction = max(0.0, min(1.0, prediction))
                
                st.markdown("### 📊 Analysis Results")
                
                col1, col2, col3 = st.columns([1,3,1])
                with col2:
                    st.progress(prediction)
                    risk_level = "🔴 High" if prediction >= 0.7 else "🟡 Medium" if prediction >= 0.4 else "🟢 Low"
                    st.markdown(f"**Risk Score:** {prediction:.1%} ({risk_level} Risk)")
                    
                    confidence = "High" if prediction >= 0.8 or prediction <= 0.2 else "Medium"
                    st.caption(f"Confidence: {confidence}")
                
                if show_details:
                    st.markdown("### 🔍 Detailed Analysis")
                    try:
                        details = get_feature_details(features)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🔍 Text Analysis")
                            with st.expander("View Details", expanded=True):
                                st.json(details["Text Analysis"])
                                st.json(details["Text Characteristics"])
                            
                        with col2:
                            st.markdown("#### 📊 Pattern Detection")
                            st.json(details["Pattern Detection"])
                                
                    except Exception as e:
                        st.warning(f"Could not generate detailed analysis: {str(e)}")
                
                # Enhanced warning system
                st.markdown("### ⚠️ Risk Assessment")
                if prediction >= 0.7:
                    st.error("🚨 **HIGH RISK**: This message shows strong indicators of suspicious content.")
                    st.markdown("**Recommended Actions:**")
                    st.markdown("- Review message context and sender")
                    st.markdown("- Consider additional verification")
                    st.markdown("- Flag for manual review")
                elif prediction >= 0.4:
                    st.warning("⚠️ **MEDIUM RISK**: This message shows some indicators of suspicious content.")
                    st.markdown("**Recommended Actions:**")
                    st.markdown("- Monitor conversation patterns")
                    st.markdown("- Consider context and sender history")
                else:
                    st.success("✅ **LOW RISK**: This message appears to be normal communication.")
                    
                st.markdown("---")
                st.markdown("### 🔬 Advanced Analytics")
                tab1, tab2, tab3 = st.tabs(["🔍 SHAP Explanation", "🧠 Similar Messages (RAG)", "🕸️ Mock Graph Analytics"])
                
                with tab1:
                    if "shap_values" in result and "shap_base_value" in result:
                        shap_vals = result["shap_values"]
                        feature_names = list(features.keys())
                        if len(shap_vals) == len(feature_names):
                            shap_df = pd.DataFrame({"Feature": feature_names, "SHAP Value": shap_vals})
                            shap_df["Absolute Impact"] = shap_df["SHAP Value"].abs()
                            shap_df = shap_df.sort_values(by="Absolute Impact", ascending=False).head(10)
                            st.bar_chart(shap_df.set_index("Feature")["SHAP Value"])
                        else:
                            st.warning("SHAP values dimension mismatch.")
                    else:
                        st.info("SHAP explanations not available for this prediction.")
                        
                with tab2:
                    try:
                        search_res = requests.post(f"{API_URL}/semantic_search", json={"query": message, "limit": 3})
                        if search_res.status_code == 200:
                            similar_msgs = search_res.json().get("results", [])
                            if similar_msgs:
                                for idx, sm in enumerate(similar_msgs):
                                    with st.expander(f"Match {idx+1} (Risk: {sm.get('risk_level', 'Unknown')})"):
                                        st.write(sm.get("message_text", ""))
                                        st.caption(f"Platform: {sm.get('platform', 'Unknown')} | Score: {sm.get('similarity_score', 0):.2f}")
                            else:
                                st.info("No similar messages found.")
                        else:
                            st.warning("Semantic search unavailable.")
                    except Exception as e:
                        st.warning("Could not retrieve similar messages.")
                        
                with tab3:
                    try:
                        # Pass a mock user_id for the current user
                        graph_res = requests.post(f"{API_URL}/network_analysis", json={"user_id": "current_user"})
                        if graph_res.status_code == 200:
                            graph_data = graph_res.json()
                            st.write(f"**Nodes:** {graph_data['metrics']['node_count']} | **Edges:** {graph_data['metrics']['edge_count']}")
                            nodes = sorted(graph_data['nodes'], key=lambda x: x.get('pagerank', 0), reverse=True)
                            st.write("Top entities in ego-network (by PageRank):")
                            st.dataframe(pd.DataFrame(nodes).head(3))
                        else:
                            st.warning("Graph analytics unavailable.")
                    except Exception as e:
                        st.warning("Could not retrieve graph analytics.")
                    
            else:
                st.error(f"Error from API (Status {response.status_code}): {response.text}")
                
        except Exception as e:
            st.error(f"Error calling API: {str(e)}")
            st.info("Please ensure the FastAPI backend is running.")

# Enhanced Footer
st.markdown("---")
st.markdown("### 📚 Additional Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔒 Privacy & Security**")
    st.markdown("""
    - Messages are processed locally
    - No data is stored permanently  
    - Input validation prevents injection
    - Secure model loading
    """)

with col2:
    st.markdown("**🎯 Accuracy Notes**")
    st.markdown("""
    - AI predictions are probabilistic
    - Context matters significantly
    - False positives/negatives possible
    - Manual verification recommended
    """)

with col3:
    st.markdown("**🛠️ Technical Details**")
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
    <p>Built with Streamlit • Powered by scikit-learn • Enhanced with custom NLP features</p>
</div>
""", unsafe_allow_html=True)
