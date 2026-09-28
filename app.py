import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Page Configuration
st.set_page_config(
    page_title="Intelligent Document Classification System", 
    page_icon="🚀", 
    layout="wide"
)

# Custom Styling / Header
st.title("🚀 Intelligent Document Classification System")
st.markdown("### Advanced NLP & Machine Learning Pipeline for Automated Text Categorization")
st.markdown("---")

# Load data and train model with caching for speed
@st.cache_resource
def load_and_train_pipeline():
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
    
    # Text Preprocessing & Feature Extraction (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(data.data)
    y = data.target
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Development (Naive Bayes Classifier)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=data.target_names, output_dict=True)
    
    return model, vectorizer, data.target_names, acc, report, data

model, vectorizer, target_names, accuracy, report, dataset = load_and_train_pipeline()

# Layout: Sidebar for navigation/info
st.sidebar.header("📊 Project Modules")
app_mode = st.sidebar.selectbox("Choose View", ["Live Prediction Portal", "Model Evaluation & Metrics", "Dataset Explorer"])

if app_mode == "Live Prediction Portal":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✍️ Test Document Classifier")
        default_text = "The graphics card and GPU performance on this computer is incredible."
        user_input = st.text_area("Enter a document, news excerpt, or support ticket:", default_text, height=150)
        
        if st.button("Run AI Classification", type="primary"):
            if not user_input.strip():
                st.warning("Please enter valid text.")
            else:
                # Transform & Predict
                X_new = vectorizer.transform([user_input])
                prediction_idx = model.predict(X_new)[0]
                probabilities = model.predict_proba(X_new)[0]
                
                predicted_category = target_names[prediction_idx]
                confidence = probabilities[prediction_idx] * 100
                
                st.success(f"**Predicted Category:** `{predicted_category}`")
                st.metric(label="Prediction Confidence Score", value=f"{confidence:.2f}%")
                
                # Show Probabilities across all categories
                st.markdown("#### 📈 Probability Breakdown Across Categories")
                prob_df = pd.DataFrame({
                    "Category": target_names,
                    "Probability (%)": probabilities * 100
                }).sort_values(by="Probability (%)", ascending=False)
                
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("💡 Quick Test Samples")
        st.markdown("Click a sample to load it:")
        if st.button("Sample 1: Computer Hardware"):
            st.session_state['sample'] = "The graphics card and GPU performance on this computer is incredible."
        if st.button("Sample 2: Medical Treatment"):
            st.session_state['sample'] = "Doctors and clinical researchers found a new treatment for the disease."
        if st.button("Sample 3: Religion / Debate"):
            st.session_state['sample'] = "Atheism and theological arguments regarding faith and science."

        st.info("This system extracts high-dimensional TF-IDF features and processes them via Multinomial Naive Bayes.")

elif app_mode == "Model Evaluation & Metrics":
    st.subheader("📈 Quantitative Model Performance")
    st.markdown("The model was evaluated on an independent test split using industry-standard metrics.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Test Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Feature Extraction", "TF-IDF Vectorizer")
    col3.metric("Classifier Algorithm", "Multinomial Naive Bayes")
    
    st.markdown("#### Detailed Classification Report (Precision, Recall, F1-Score)")
    # Convert classification report dictionary to a clean dataframe for display
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

elif app_mode == "Dataset Explorer":
    st.subheader("📁 Training Dataset Overview")
    st.markdown(f"Total documents loaded in background engine: **{len(dataset.data)}** across **{len(target_names)}** target classes.")
    
    st.markdown("#### Available Target Classes:")
    for idx, name in enumerate(target_names):
        st.markdown(f"- **Class {idx}:** `{name}`")
    
    selected_class = st.selectbox("View sample document from class:", target_names)
    class_idx = list(target_names).index(selected_class)
    
    # Find a sample matching this class
    sample_texts = [dataset.data[i] for i, target in enumerate(dataset.target) if target == class_idx]
    if sample_texts:
        st.text_area(f"Raw Document Sample from '{selected_class}':", sample_texts[0][:1000] + "...", height=200)