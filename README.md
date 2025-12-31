# Word-Etymology-Predictor

A machine learning-powered application that predicts the origin language and relationship type of any word. Uses advanced models to analyze character patterns and linguistic features to determine whether a word was borrowed, derived, or inherited from its origin language.

## Features

- **Dual ML Models**: Compare predictions between Logistic Regression and Deep Neural Networks
- **Relationship Type Detection**: Identifies if a word is borrowed_from, derived_from, or inherited_from its origin
- **Real-time Model Switching**: Instantly switch between models to see prediction differences
- **Interactive Web UI**: Clean, responsive Streamlit interface
- **Confidence Scores**: See probability distribution across all supported languages

## Models

### 1. Logistic Regression (Fast & Reliable)
- **Features**: TF-IDF character n-grams (1-4 characters) + encoded reltype
- **Approach**: Statistical pattern matching using term frequency analysis
- **Speed**: Instant predictions
- **Strength**: Excellent with Latin-based scripts, predictable and interpretable

### 2. CNN (Convolutional Neural Network)
- **Architecture**: 
  - Word branch: Character embedding (128-dim) → 2x Conv1D layers → Global max pooling
  - Reltype branch: One-hot encoding → Dense layer
  - Fusion: Concatenated branches → Classification layers
- **Features**: Character sequences + relationship type context
- **Speed**: Fast predictions
- **Strength**: Detects complex patterns, handles diverse scripts (Arabic, Greek, Hindi)

## Dataset

**File**: `word_origin_dataset_cleaned.csv`

### Columns
- `word`: The word to analyze
- `origin_language`: The source language (target variable)
- `reltype`: Relationship type between word and origin

### Relationship Types
- `borrowed_from`: Word adopted from another language (e.g., "pizza" from Italian)
- `derived_from`: Word created from another language's root (e.g., "astronomy" from Greek)
- `inherited_from`: Word passed down from ancient language family (e.g., "mother" from Proto-Indo-European)

### Supported Languages
Multiple language origins including: Latin, Greek, Arabic, French, German, Italian, Hindi, Spanish, and more.

## Installation

### Requirements
- Python 3.8+
- TensorFlow 2.x
- Streamlit
- scikit-learn
- pandas
- numpy

### Setup
```bash
# Clone/download the repository
cd Word-Etymology-Predictor

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Training the Models

```bash
python train_models.py
```

This will:
- Load and prepare the dataset
- Train the Logistic Regression model with TF-IDF + reltype features
- Train the CNN model with character embedding + reltype dual-input architecture
- Save trained models and encoders:
  - `logreg_model.pkl` - Logistic Regression model
  - `tfidf_vectorizer.pkl` - TF-IDF vectorizer
  - `reltype_encoder.pkl` - Reltype encoder for LogReg
  - `cnn_model.h5` - Trained CNN model
  - `char_mapping.json` - Character mappings and metadata
  - `reltype_cnn_encoder.pkl` - Reltype encoder for CNN
  - `model_accuracy.json` - Accuracy metrics

**Training Output**:
- Logistic Regression Accuracy (on test set)
- CNN Accuracy (on test set)
- Distribution of training data

### 2. Running the Application

```bash
streamlit run app.py
```

The app will start on `http://localhost:8501`

### 3. Using the Web Interface

1. **Enter a Word**: Type any word in the input field
   - Examples: "safari", "pizza", "tsunami", "algebra"

2. **Select Relationship Type**: Choose from dropdown (default: "borrowed_from")
   - `borrowed_from`: Word adopted directly
   - `derived_from`: Word created from origin root
   - `inherited_from`: Word from ancestral language

3. **Choose Model**: Select which model to use
   - **Logistic Regression**: Fast, reliable, good for familiar patterns
   - **CNN**: Better for complex patterns, diverse scripts

4. **Click Predict**: Models analyze the word and return results

5. **View Results**:
   - Top prediction with confidence score
   - All language predictions with probability bars
   - Compare instantly by switching models or relationship type

### 4. Viewing Statistics

Right sidebar displays:
- **Dataset Stats**: Total words and number of supported languages
- **Model Accuracy**: Accuracy of both models on test set
- **Test Set Size**: Number of samples used for evaluation

