# RelType Implementation - Word Origin Predictor

## Overview
Added relationship type (reltype) feature to both Logistic Regression and CNN models. Models now consider whether a word is **borrowed_from**, **derived_from**, or **inherited_from** when predicting origin language.

## Changes Made

### 1. Training Script (`train_models.py`)

#### Data Processing
- Extract and validate `reltype` column from dataset
- Split training/test data preserving reltype information
- Create separate encoders for reltype (one for LogReg, one for CNN)

#### Logistic Regression Model
- **Feature Engineering**: Combine TF-IDF character n-grams with encoded reltype
- **Input**: Concatenate TF-IDF sparse matrix (7000 features) with reltype categorical feature
- **Benefit**: Linguistic patterns (n-grams) + relationship type context

#### CNN Model Architecture (Dual Input)
- **Word Branch**:
  - Character embedding (128-dim)
  - 2x Conv1D layers (128 filters, kernel_size=3)
  - Global max pooling
  - Dropout (0.5)

- **Reltype Branch**:
  - One-hot encoded reltype input
  - Dense layer (32 units, relu)
  - Dropout (0.3)

- **Fusion**: Concatenate both branches
- **Classification**: Dense layers → Softmax output

#### Saved Artifacts
- `reltype_encoder.pkl` - LogReg reltype encoder
- `reltype_cnn_encoder.pkl` - CNN reltype encoder
- `char_mapping.json` - Updated with reltype classes list

### 2. Prediction Functions (`app.py`)

#### Logistic Regression Prediction
```python
def predict_lr(word, reltype=None):
    - Transform word to TF-IDF features
    - If reltype provided: encode and concatenate
    - Return ranked language predictions
```

#### CNN Prediction
```python
def predict_cnn_model(word, reltype=None):
    - Encode word characters, pad to 50 chars
    - If reltype provided: one-hot encode and pass to reltype_input
    - Otherwise: pass zero vector (model gracefully handles missing reltype)
    - Return ranked language predictions
```

### 3. User Interface (`app.py`)

#### New Control
- **Reltype Dropdown**: Optional selector with values:
  - "" (empty/unknown)
  - "borrowed_from"
  - "derived_from"
  - "inherited_from"
- Positioned between Model selector and Predict button
- Help text: "Helps refine predictions if you know the relationship type"

#### Dynamic Updates
- Model switching with reltype updates predictions instantly
- Reltype changes trigger re-evaluation
- Can predict with or without reltype (optional feature)

## Usage

### Training
```bash
python train_models.py
# Trains both models considering reltype
# Saves encoders and updated metadata
```

### Prediction
```python
# Without reltype (uses default zero vector)
results = predict_lr("pizza")

# With reltype
results = predict_lr("pizza", reltype="borrowed_from")
results = predict_cnn_model("tsunami", reltype="inherited_from")
```

### In App
1. Enter word (e.g., "pizza")
2. Select relationship type (e.g., "borrowed_from") - optional
3. Select model
4. Click "Predict" → See ranked results

## Benefits

1. **Better Accuracy**: Models learn that certain language-reltype combinations are more likely
2. **Linguistic Context**: Captures relationship types (borrowed words behave differently from inherited ones)
3. **Flexible**: Reltype is optional - app works fine without it
4. **Model Comparison**: Easy to see how reltype affects different models

## Example
- Word: "pizza"
- Known reltype: "borrowed_from"
- Models will weight Italian origin higher when reltype=borrowed_from
- Different prediction confidence compared to prediction without reltype info

## Files Modified
- `train_models.py` - Core training logic
- `app.py` - Prediction functions and UI
- Dataset: `word_origin_dataset_cleaned.csv` (used as-is with reltype column)

## Files Created
- `reltype_encoder.pkl` - LogReg reltype encoder
- `reltype_cnn_encoder.pkl` - CNN reltype encoder
