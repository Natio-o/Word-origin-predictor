import pandas as pd
import numpy as np
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Input, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def train():
    print("Grabbing the cleaned data...")
    try:
        df = pd.read_csv('word_origin_dataset_cleaned.csv')
    except Exception as e:
        print(f"Couldn't load the file: {e}")
        return

    # Just a quick check to make sure there's no empty stuff in the word column.
    df = df.dropna(subset=['word', 'origin_language', 'reltype'])
    print(f"Total words to train on: {len(df)}")
    print(f"\nLanguage Distribution:")
    print(df['origin_language'].value_counts())
    print(f"\nRelationship Type Distribution:")
    print(df['reltype'].value_counts())

    # --- MODEL 1: Logistic Regression ---
    print("\nStarting on the Logistic Regression model...")
    words = df['word'].astype(str).tolist()
    labels = df['origin_language'].tolist()
    reltypes = df['reltype'].tolist()

    # We'll set aside 20% of the data to test the model afterward.
    X_train_words, X_test_words, y_train, y_test, rel_train, rel_test = train_test_split(
        words, labels, reltypes, test_size=0.3, random_state=42, stratify=labels
    )

    # Encode reltype as categorical feature
    reltype_encoder = LabelEncoder()
    rel_train_encoded = reltype_encoder.fit_transform(rel_train)
    rel_test_encoded = reltype_encoder.transform(rel_test)
    
    # Save reltype encoder
    with open('reltype_encoder.pkl', 'wb') as f:
        pickle.dump(reltype_encoder, f)

    # Turning words into numbers using TF-IDF. 
    # We're looking at patterns of 1 to 4 characters (n-grams).
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,4), max_features=7000)
    X_train_tfidf = vectorizer.fit_transform(X_train_words)
    X_test_tfidf = vectorizer.transform(X_test_words)
    
    # Combine TF-IDF features with reltype features
    # Convert sparse matrix to dense and append reltype as a column
    X_train_combined = np.hstack([X_train_tfidf.toarray(), rel_train_encoded.reshape(-1, 1)])
    X_test_combined = np.hstack([X_test_tfidf.toarray(), rel_test_encoded.reshape(-1, 1)])

    # Balancing weights so the model pays equal attention to all languages.
    classes = np.unique(y_train)
    cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, cw))

    # Setting up the model. We're using 'lbfgs' because it's fast and reliable.
    logreg = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', 
                                class_weight=class_weights_dict, C=1.0, n_jobs=-1)
    logreg.fit(X_train_combined, y_train)

    # Let's see how accurate it is.
    logreg_accuracy = logreg.score(X_test_combined, y_test)
    print(f"LogReg Accuracy (with reltype): {logreg_accuracy*100:.2f}%")

    # Saving the model and its vectorizer so the app can use them later.
    with open('logreg_model.pkl', 'wb') as f:
        pickle.dump(logreg, f)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # --- MODEL 2: CNN (Deep Learning) ---
    print("\nNow training the CNN model...")
    # Creating a map of every character we've seen to a unique number.
    chars = sorted(list(set("".join(X_train_words))))
    char_to_int = {c: i+1 for i, c in enumerate(chars)}
    max_len = 50 # Standardizing word length to 50 characters.

    def encode_text(w, char_map, length):
        return [char_map.get(c, 0) for c in w]

    # Prepping the data for the neural network.
    X_train_cnn = [encode_text(w, char_to_int, max_len) for w in X_train_words]
    X_train_cnn = pad_sequences(X_train_cnn, maxlen=max_len, padding='post')
    
    X_test_cnn = [encode_text(w, char_to_int, max_len) for w in X_test_words]
    X_test_cnn = pad_sequences(X_test_cnn, maxlen=max_len, padding='post')
    
    # Prepare reltype features for CNN (one-hot encode for better neural network usage)
    reltype_cnn_encoder = LabelEncoder()
    rel_train_cnn = reltype_cnn_encoder.fit_transform(rel_train)
    rel_test_cnn = reltype_cnn_encoder.transform(rel_test)
    
    # One-hot encode reltype
    rel_train_onehot = to_categorical(rel_train_cnn, num_classes=len(reltype_cnn_encoder.classes_))
    rel_test_onehot = to_categorical(rel_test_cnn, num_classes=len(reltype_cnn_encoder.classes_))

    # Encoding labels (language names) into numbers.
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_train_categorical = to_categorical(y_train_encoded)
    y_test_encoded = le.transform(y_test)
    y_test_categorical = to_categorical(y_test_encoded)

    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
    class_weights = dict(enumerate(weights))

    vocab_size = len(char_to_int) + 1
    num_classes = len(le.classes_)
    num_reltypes = len(reltype_cnn_encoder.classes_)

    # Building the architecture with dual inputs (words + reltype).
    # Word branch: Embedding -> Convolutions -> Global Max Pooling
    # Reltype branch: Direct input for relationship type
    # Both branches merge and feed through dense layers.
    from tensorflow.keras.layers import Input, Concatenate
    from tensorflow.keras.models import Model
    
    # Word input branch
    word_input = Input(shape=(max_len,), name='word_input')
    word_embed = Embedding(input_dim=vocab_size, output_dim=128)(word_input)
    word_conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(word_embed)
    word_bn1 = BatchNormalization()(word_conv1)
    word_conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(word_bn1)
    word_pool = GlobalMaxPooling1D()(word_conv2)
    word_drop = Dropout(0.5)(word_pool)
    
    # Reltype input branch
    reltype_input = Input(shape=(num_reltypes,), name='reltype_input')
    reltype_dense = Dense(32, activation='relu')(reltype_input)
    reltype_drop = Dropout(0.3)(reltype_dense)
    
    # Concatenate both branches
    merged = Concatenate()([word_drop, reltype_drop])
    
    # Dense layers after merge
    dense1 = Dense(128, activation='relu')(merged)
    bn = BatchNormalization()(dense1)
    drop1 = Dropout(0.3)(bn)
    dense2 = Dense(64, activation='relu')(drop1)
    drop2 = Dropout(0.2)(dense2)
    output = Dense(num_classes, activation='softmax')(drop2)
    
    # Create model with dual inputs
    model = Model(inputs=[word_input, reltype_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training logic with a safety brake: it stops if it doesn't improve for 3 turns.
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001, verbose=1)
    
    model.fit([X_train_cnn, rel_train_onehot], y_train_categorical, epochs=30, batch_size=128, validation_split=0.15, 
              class_weight=class_weights, callbacks=[early_stop, reduce_lr], verbose=1)

    # Check the final score.
    _, cnn_accuracy = model.evaluate([X_test_cnn, rel_test_onehot], y_test_categorical, verbose=0)
    print(f"CNN Accuracy (with reltype): {cnn_accuracy*100:.2f}%")

    # Save the model and the character map.
    model.save('cnn_model.h5')
    
    # Save reltype encoder for CNN
    with open('reltype_cnn_encoder.pkl', 'wb') as f:
        pickle.dump(reltype_cnn_encoder, f)
    
    meta_data = {
        "char_to_int": char_to_int,
        "max_len": max_len,
        "labels": le.classes_.tolist(),
        "reltypes": reltype_cnn_encoder.classes_.tolist()
    }
    with open('char_mapping.json', 'w') as f:
        json.dump(meta_data, f)
    
    # Writing down the accuracy scores so the app can display them.
    accuracy_metrics = {
        "logistic_regression_accuracy": float(logreg_accuracy),
        "cnn_accuracy": float(cnn_accuracy),
        "test_samples": len(X_test_words),
        "train_samples": len(X_train_words)
    }
    with open('model_accuracy.json', 'w') as f:
        json.dump(accuracy_metrics, f, indent=2)
    
    print("\nAll done! Both models are saved and ready.")

if __name__ == "__main__":
    train()
