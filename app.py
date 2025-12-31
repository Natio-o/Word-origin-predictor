import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Setup
st.set_page_config(
    page_title="Word Origin Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');
    
    * {
        font-family: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    html, body, [class*="css"] {
        font-family: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        color: #1a1a1a;
    }

    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        color: #0a0a0a !important;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }

    h3 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #2a2a2a !important;
    }

    p, li, span, label {
        color: #4a4a4a !important;
        line-height: 1.6 !important;
        font-size: 0.95rem !important;
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.1);
    }

    .hero-container h1 {
        color: white !important;
        margin: 0 !important;
        font-size: 3rem !important;
    }

    .hero-container p {
        color: rgba(255,255,255,0.8) !important;
        font-size: 1.1rem !important;
        margin-top: 0.75rem !important;
    }
    


    /* Result Card */
    .result-card {
        background: white;
        padding: 3rem;
        border-radius: 16px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.08);
        margin-top: 2rem;
        border: 1px solid #e5e7eb;
    }
    
    .top-result-lang {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0a7ea4 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0;
        letter-spacing: -1px;
    }
    
    .top-result-prob {
        font-size: 1.15rem;
        font-weight: 600;
        color: white;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 0.6rem 1.5rem;
        border-radius: 50px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }

    /* Stats Section */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        text-align: center;
    }

    .stat-card h3 {
        margin: 0 !important;
        font-size: 0.9rem !important;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem !important;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0a0a0a;
        margin: 0;
    }

    /* Model Info Cards */
    .model-card {
        background: white;
        padding: 1.75rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .model-card h4 {
        color: #0a0a0a !important;
        font-size: 1.15rem !important;
        margin: 0 0 1rem 0 !important;
        font-weight: 600 !important;
    }

    /* Comparison Table */
    .comparison-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 2rem 0;
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    
    .comparison-table th, .comparison-table td {
        padding: 1rem 1.25rem;
        text-align: left;
        border-bottom: 1px solid #e5e7eb;
        color: #2a2a2a;
    }
    
    .comparison-table th {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        color: #1a1a1a;
        font-weight: 600;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    .comparison-table tr:last-child td {
        border-bottom: none;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: #e5e7eb;
        margin: 2.5rem 0;
    }

    /* Stacked on Mobile */
    @media (max-width: 768px) {
        .stApp {
            padding: 1rem;
        }

        h1 {
            font-size: 2rem !important;
        }

        h2 {
            font-size: 1.5rem !important;
        }

        .hero-container {
            padding: 2rem 1.5rem;
        }

        .hero-container h1 {
            font-size: 2.25rem !important;
        }

        .input-section {
            padding: 1.5rem;
        }

        .result-card {
            padding: 2rem 1.5rem;
        }

        .top-result-lang {
            font-size: 3rem;
        }

        .comparison-table th, .comparison-table td {
            padding: 0.75rem;
            font-size: 0.85rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Resource Loading ---

@st.cache_resource
def load_logreg():
    try:
        if not os.path.exists('logreg_model.pkl'): return None, None, None
        with open('logreg_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vect = pickle.load(f)
        with open('reltype_encoder.pkl', 'rb') as f:
            reltype_enc = pickle.load(f)
        return model, vect, reltype_enc
    except: return None, None, None

@st.cache_resource
def load_cnn():
    try:
        if not os.path.exists('cnn_model.h5'): return None, None, None
        model = load_model('cnn_model.h5')
        with open('char_mapping.json', 'r') as f:
            meta = json.load(f)
        with open('reltype_cnn_encoder.pkl', 'rb') as f:
            reltype_enc = pickle.load(f)
        return model, meta, reltype_enc
    except: return None, None, None

@st.cache_data
def get_stats():
    try:
        path = 'word_origin_dataset_cleaned.csv'
        if not os.path.exists(path): return 0, []
        df = pd.read_csv(path)
        return len(df), sorted(df['origin_language'].unique().tolist())
    except: return 0, []

@st.cache_data
def get_accuracy():
    try:
        if not os.path.exists('model_accuracy.json'): return None
        with open('model_accuracy.json', 'r') as f:
            return json.load(f)
    except: return None

# Load everything
logreg_model, logreg_vect, logreg_reltype_enc = load_logreg()
cnn_model, cnn_meta, cnn_reltype_enc = load_cnn()
total_words, languages = get_stats()
accuracy = get_accuracy()

# --- Prediction Logic ---

def predict_lr(word, reltype=None):
    """Predict using Logistic Regression with optional reltype"""
    X = logreg_vect.transform([word])
    
    # Add reltype feature if provided
    if reltype and logreg_reltype_enc:
        try:
            reltype_encoded = logreg_reltype_enc.transform([reltype])[0]
            X_combined = np.hstack([X.toarray(), np.array([[reltype_encoded]])])
            probs = logreg_model.predict_proba(X_combined)[0]
        except:
            probs = logreg_model.predict_proba(X)[0]
    else:
        probs = logreg_model.predict_proba(X)[0]
    
    res = [{"lang": c, "prob": p} for c, p in zip(logreg_model.classes_, probs)]
    return sorted(res, key=lambda x: x['prob'], reverse=True)

def predict_cnn_model(word, reltype=None):
    """Predict using CNN with optional reltype"""
    char_map = cnn_meta['char_to_int']
    max_len = cnn_meta['max_len']
    labels = cnn_meta['labels']
    
    enc = [char_map.get(c, 0) for c in word]
    padded = pad_sequences([enc], maxlen=max_len, padding='post')
    
    # Prepare reltype feature
    if reltype and cnn_reltype_enc:
        try:
            reltype_encoded = cnn_reltype_enc.transform([reltype])[0]
            # One-hot encode reltype
            num_reltypes = len(cnn_reltype_enc.classes_)
            reltype_onehot = np.zeros((1, num_reltypes))
            reltype_onehot[0, reltype_encoded] = 1
            probs = cnn_model.predict([padded, reltype_onehot], verbose=0)[0]
        except:
            probs = cnn_model.predict([padded, np.zeros((1, len(cnn_reltype_enc.classes_)))], verbose=0)[0]
    else:
        # Default: use zero-vector for reltype
        num_reltypes = len(cnn_reltype_enc.classes_) if cnn_reltype_enc else 3
        probs = cnn_model.predict([padded, np.zeros((1, num_reltypes))], verbose=0)[0]
    
    res = [{"lang": l, "prob": p} for l, p in zip(labels, probs)]
    return sorted(res, key=lambda x: x['prob'], reverse=True)

# --- UI Layout ---

# Hero Section
st.markdown("""
<div class='hero-container'>
    <h1>Word Origin Predictor</h1>
    <p>Discover the linguistic roots of any word using AI-powered machine learning models</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for word input
if "current_word" not in st.session_state:
    st.session_state.current_word = ""
if "current_results" not in st.session_state:
    st.session_state.current_results = None

# Main Content Layout
col_main, col_sidebar = st.columns([3, 1], gap="large")

with col_main:
    # Input Section with clean styling
    word_input = st.text_input(
        "Enter a word to analyze",
        placeholder="e.g. 'safari', 'pizza', or 'tsunami'",
        label_visibility="collapsed",
        key="word_input_field"
    )
    
    col_model, col_reltype, col_btn = st.columns([1.5, 1.5, 1], gap="medium")
    with col_model:
        selected_model = st.selectbox(
            "Select Model",
            ["Logistic Regression (Fast & Reliable)", "CNN (Deep Neural Network)"],
            label_visibility="collapsed",
            key="model_selector"
        )
    
    with col_reltype:
        selected_reltype = st.selectbox(
            "Relationship Type",
            ["borrowed_from", "derived_from", "inherited_from"],
            index=0,
            label_visibility="collapsed",
            key="reltype_selector",
            help="Choose the relationship type between the word and its origin"
        )
    
    with col_btn:
        predict_btn = st.button("Predict", type="primary", use_container_width=True)
    
    # Results Section with dynamic updates
    if predict_btn and word_input:
        st.session_state.current_word = word_input.lower().strip()
    
    # Show results if we have a word (either from button or from switching models)
    if st.session_state.current_word:
        with st.spinner("🔍 Analyzing patterns..."):
            if "Logistic" in selected_model:
                results = predict_lr(st.session_state.current_word, reltype=selected_reltype) if logreg_model else []
            else:
                results = predict_cnn_model(st.session_state.current_word, reltype=selected_reltype) if cnn_model else []
            
            if results:
                st.session_state.current_results = results
                
                top = results[0]
                st.markdown(f"""
                <div class='result-card'>
                    <div style='text-align: center; padding: 1rem 0;'>
                        <p style='color: #9ca3af; text-transform: uppercase; font-weight: 500; letter-spacing: 0.5px; margin: 0; font-size: 0.9rem;'>Predicted Origin</p>
                        <h2 class='top-result-lang'>{top['lang'].upper()}</h2>
                        <span class='top-result-prob'>{top['prob']*100:.1f}% Confidence</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### All Predictions", help="Full confidence scores across all languages")
                
                # Create prediction bars with better styling
                for idx, res in enumerate(results):
                    col_lang, col_bar = st.columns([0.8, 3.2], gap="small")
                    with col_lang:
                        st.markdown(f"<strong style='font-size: 0.95rem;'>{res['lang']}</strong>", unsafe_allow_html=True)
                    with col_bar:
                        st.progress(float(res['prob']), text=f"{res['prob']*100:.1f}%")
            elif st.session_state.current_word:
                st.error("⚠️ Model files not found. Please train the models first.")

with col_sidebar:
    st.markdown("### 📊 Stats")
    
    if total_words > 0:
        st.markdown(f"""
        <div class="stat-card">
            <h3>Dataset</h3>
            <p class="stat-value">{total_words:,}</p>
            <p style='margin: 0; font-size: 0.85rem; color: #6b7280;'>Words</p>
        </div>
        """, unsafe_allow_html=True)
        
        langs_display = f"{len(languages)}" if languages else "0"
        st.markdown(f"""
        <div class="stat-card">
            <h3>Languages</h3>
            <p class="stat-value">{langs_display}</p>
            <p style='margin: 0; font-size: 0.85rem; color: #6b7280;'>Supported</p>
        </div>
        """, unsafe_allow_html=True)
    
    if accuracy:
        lr_acc = accuracy['logistic_regression_accuracy'] * 100
        cnn_acc = accuracy['cnn_accuracy'] * 100
        
        st.markdown("### 🎯 Accuracy")
        st.markdown(f"""
        <div class="stat-card">
            <h3>LogReg</h3>
            <p class="stat-value">{lr_acc:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-card">
            <h3>CNN</h3>
            <p class="stat-value">{cnn_acc:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"Test set: {accuracy['test_samples']:,} samples")

st.divider()

# --- Comparison Section ---

st.header("🧠 Model Comparison")

col_lr, col_cnn = st.columns(2, gap="medium")

with col_lr:
    st.markdown("""
    <div class="model-card">
        <h4>📊 Logistic Regression</h4>
        <p><strong>How it works:</strong></p>
        <p>Uses TF-IDF to identify unique character sequences called "n-grams". It learns which groups of letters are distinctive to each language.</p>
        <p><strong>Strengths:</strong></p>
        <ul style="margin: 0.5rem 0;">
            <li>Fast predictions</li>
            <li>Predictable and transparent</li>
            <li>Excellent with Latin-based scripts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_cnn:
    st.markdown("""
    <div class="model-card">
        <h4>🧠 CNN (Deep Neural Network)</h4>
        <p><strong>How it works:</strong></p>
        <p>Slides convolutional filters across characters to detect patterns. Learns that letter sequences matter contextually.</p>
        <p><strong>Strengths:</strong></p>
        <ul style="margin: 0.5rem 0;">
            <li>Detects complex patterns</li>
            <li>Handles diverse scripts</li>
            <li>Learns hierarchical features</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.write("### Quick Comparison Summary")
st.markdown("""
<table class="comparison-table">
    <thead>
        <tr>
            <th>Feature</th>
            <th>Logistic Regression</th>
            <th>CNN Neural Network</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Strength</strong></td>
            <td>Detecting fixed character patterns (n-grams)</td>
            <td>Detecting complex, non-linear relationships</td>
        </tr>
        <tr>
            <td><strong>Training Speed</strong></td>
            <td>Very Fast (Seconds)</td>
            <td>Moderate (Minutes)</td>
        </tr>
        <tr>
            <td><strong>Prediction Speed</strong></td>
            <td>Instant</td>
            <td>Fast</td>
        </tr>
        <tr>
            <td><strong>Best Case</strong></td>
            <td>Latin-script based languages</td>
            <td>Diverse scripts (Arabic, Greek, Hindi)</td>
        </tr>
    </tbody>
</table>
""", unsafe_allow_html=True)

# --- Detailed Explanation Section ---

st.divider()
st.header("📖 How It Works")

with st.expander("🧠 Model Architecture & Design", expanded=False):
    col_arch1, col_arch2 = st.columns(2, gap="medium")
    
    with col_arch1:
        st.markdown("""
        <div class="model-card">
            <h4>Logistic Regression</h4>
            <p><strong>Concept:</strong> Statistical pattern matching using TF-IDF vectorization.</p>
            <p><strong>Process:</strong></p>
            <ul>
                <li>Breaks words into 1-4 character "n-grams"</li>
                <li>Learns which n-grams are unique to each language</li>
                <li>Assigns confidence scores based on n-gram frequency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_arch2:
        st.markdown("""
        <div class="model-card">
            <h4>Convolutional Neural Network</h4>
            <p><strong>Concept:</strong> Deep learning pattern recognition, similar to image processing.</p>
            <p><strong>Process:</strong></p>
            <ul>
                <li>Embeds each character in 128-dimensional space</li>
                <li>Slides convolutional filters to detect patterns</li>
                <li>Learns hierarchical linguistic features</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with st.expander("📊 Training & Evaluation", expanded=False):
    col_train1, col_train2 = st.columns(2, gap="medium")
    
    with col_train1:
        st.markdown("""
        <div class="model-card">
            <h4>Data Preparation</h4>
            <ul>
                <li><strong>80/20 Split:</strong> 80% training, 20% blind test</li>
                <li><strong>LabelEncoder:</strong> Maps languages to numeric IDs</li>
                <li><strong>Class Weights:</strong> Balances small language representation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_train2:
        st.markdown("""
        <div class="model-card">
            <h4>Evaluation Metrics</h4>
            <ul>
                <li><strong>Test Set:</strong> Unseen during training</li>
                <li><strong>Accuracy:</strong> Exact language prediction match</li>
                <li><strong>Metrics:</strong> Persisted in <code>model_accuracy.json</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with st.expander("⚡ Prediction Pipeline", expanded=False):
    st.markdown("""
    <div class="model-card">
        <ol>
            <li><strong>Input Normalization:</strong> Convert word to lowercase</li>
            <li><strong>Feature Transformation:</strong> 
                <ul style="margin: 0.5rem 0;">
                    <li>LogReg: Extract n-grams using TF-IDF vectorizer</li>
                    <li>CNN: Pad/encode characters to fixed length (50)</li>
                </ul>
            </li>
            <li><strong>Model Prediction:</strong> Pass through trained model</li>
            <li><strong>Softmax Activation:</strong> Convert scores to probabilities (sum = 100%)</li>
            <li><strong>Results:</strong> Return ranked language predictions</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
