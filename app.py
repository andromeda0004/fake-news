import streamlit as st
import joblib
import pandas as pd
import string
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Title and description
st.title("📰 Fake News Detection System")
st.markdown("""
This application uses machine learning to detect whether a news article is **fake** or **real**.
Simply paste the text of a news article below and click **Analyze** to get the prediction.
""")

# Load the pre-trained model
@st.cache_resource
def load_model():
    model_path = Path("fake_news_model.joblib")
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Text preprocessing function (matching the notebook preprocessing)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    return text

# Load model
model = load_model()

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter News Article Text")
    
    # Text input area
    article_text = st.text_area(
        "Paste the article text here:",
        height=300,
        placeholder="Enter or paste the news article text you want to analyze..."
    )
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze Article", type="primary", use_container_width=True)
    
    if analyze_button:
        if not article_text or article_text.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing the article..."):
                # Preprocess the text
                processed_text = preprocess_text(article_text)
                
                # Make prediction
                prediction = model.predict([processed_text])[0]
                prediction_proba = model.predict_proba([processed_text])
                
                # Get confidence scores
                fake_confidence = prediction_proba[0][0] * 100
                real_confidence = prediction_proba[0][1] * 100
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Prediction result with colored box
                if prediction == "fake":
                    st.error(f"### 🚨 Prediction: FAKE NEWS")
                    st.metric("Fake Confidence", f"{fake_confidence:.2f}%")
                else:
                    st.success(f"### ✅ Prediction: REAL NEWS")
                    st.metric("Real Confidence", f"{real_confidence:.2f}%")
                
                # Show confidence breakdown
                st.markdown("#### Confidence Breakdown:")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Fake", f"{fake_confidence:.2f}%")
                with col_b:
                    st.metric("Real", f"{real_confidence:.2f}%")
                
                # Progress bars for visual representation
                st.markdown("#### Visual Confidence:")
                st.progress(fake_confidence / 100, text=f"Fake: {fake_confidence:.2f}%")
                st.progress(real_confidence / 100, text=f"Real: {real_confidence:.2f}%")

with col2:
    st.subheader("ℹ️ About")
    st.info("""
    **How it works:**
    
    This model was trained on thousands of news articles to distinguish between fake and real news.
    
    The model uses:
    - Text vectorization (CountVectorizer)
    - TF-IDF transformation
    - Machine Learning classification
    
    **Preprocessing steps:**
    1. Convert text to lowercase
    2. Remove punctuation
    3. Remove stopwords
    4. Vectorize and transform
    5. Classify
    """)
    
    st.subheader("📝 Example Usage")
    st.markdown("""
    **Tips for best results:**
    - Paste the full article text
    - Include complete sentences
    - Avoid very short snippets
    - The longer the text, the more accurate the prediction
    """)
    
    st.subheader("⚙️ Model Info")
    st.markdown(f"""
    - **Model Type:** Pipeline with Logistic Regression
    - **Status:** ✅ Loaded successfully
    - **Version:** Pre-trained model
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Fake News Detection System | Built with Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
