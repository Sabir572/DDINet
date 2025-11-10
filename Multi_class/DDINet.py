import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (confusion_matrix, recall_score, f1_score, precision_score,
                             matthews_corrcoef, roc_auc_score, precision_recall_curve, auc)
from sklearn.preprocessing import label_binarize
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress Warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

#%%
# Set GPU configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Modify based on available GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
print("Num GPUs Available: ", len(gpus))

#%% Data loading function
def load_data(path, file_name):
    full_path = os.path.join(path, file_name)
    data = np.load(full_path, allow_pickle=True)
    return data.reshape(data.shape[0], -1).astype("float32")  # Ensure float32 for memory efficiency

def load_separate_labels(path, labels_file):
    return np.load(os.path.join(path, labels_file))

def load_combined_npy(path, file_name):
    data = load_data(path, file_name)
    y = data[:, -1]
    X = data[:, :-1]
    return X, y

# Define paths for all scenarios
SCENARIOS = {
    'S1': {  # Random split from seen_seen DDI
        'type': 'split',
        'path': "/drug_project/multi_class/seen_seen DDI/",
        'feature_file': "concat_Morgan.npy",
        'label_file': "labels.npy"
    },
    'S2': {  # Seen-unseen
        'type': 'separate',
        'train_path': "/drug_project/multi_class/seen_unseen train/",
        'test_path': "/drug_project/multi_class/s2/",
        'feature_file': "concat_Morgan.npy",
        'label_file': "labels.npy"
    },
    'S3': {  # Another split
        'type': 'separate',
        'train_path': "/drug_project/multi_class/seen_unseen train/",
        'test_path': "/drug_project/multi_class/s3/",
        'feature_file': "concat_Morgan.npy",
        'label_file': "labels.npy"
    },
    'Scaffold': {  # Scaffold splitting
        'type': 'combined',
        'train_path': "/multi_class/scaffold_splitting/",
        'test_path': "/drug_project/multi_class/scaffold_splitting/",
        'train_file': "scaffold_train_morgan_fn.npy",
        'test_file': "scaffold_test_morgan_fn.npy"
    }
}

#%%
# Model definition
def my_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Dense(436, activation='relu'),
        Dense(256, activation='relu', kernel_regularizer=l2(0.003524291993138782)),
        Dropout(0.3326933041901533),
        Dense(128, activation='relu', kernel_regularizer=l2(0.003524291993138782)),
        BatchNormalization(),
        Dropout(0.241),
        Dense(128, activation='relu', kernel_regularizer=l2(0.003524291993138782)),
        Dropout(0.3326933041901533),
        Dense(100, activation='relu', kernel_regularizer=l2(0.003524291993138782)),
        BatchNormalization(),
        Dropout(0.241),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#%% Train and validate the model with Stratified K-Fold
def train_and_validate(X_train, y_train_categorical, scenario):
    start_time = time.time()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    history_by_fold = []
    validation_accuracies = []
    validation_losses = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, np.argmax(y_train_categorical, axis=1))):
        print(f"Training fold {fold + 1} for {scenario}")
        model = my_model(input_shape=(X_train.shape[1],), num_classes=106)
        param_count = model.count_params()
        checkpoint_path = f"{scenario}/morgan_{scenario}_{fold + 1}.h5"
        callbacks = [
            ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=15)
        ]
        history = model.fit(
            X_train[train_idx], y_train_categorical[train_idx],
            epochs=80, batch_size=256,
            validation_data=(X_train[val_idx], y_train_categorical[val_idx]),
            callbacks=callbacks
        )
        models.append(model)
        history_by_fold.append(history)
        validation_accuracies.append(history.history['val_accuracy'][-1])
        validation_losses.append(history.history['val_loss'][-1])
    train_time = time.time() - start_time
    
    # Peak VRAM approximation (check after training)
    if gpus:
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        peak_vram_gb = memory_info['peak'] / (1024 ** 3) if 'peak' in memory_info else 0
    else:
        peak_vram_gb = 0
    
    print(f"{scenario} - Params: {param_count}, Train Time: {train_time:.2f}s, Peak VRAM: {peak_vram_gb:.2f}GB")
    return models, history_by_fold, param_count, train_time, peak_vram_gb

#%% Compute detailed metrics (updated with PR macro/micro)
def compute_detailed_metrics(y_true, y_pred, y_pred_prob, num_classes):
    # Precision and Recall first
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    tn = np.diag(cm)
    fp = cm.sum(axis=0) - tn
    fn = cm.sum(axis=1) - tn
    tp = cm.sum() - (fp + fn + tn)
    epsilon = 1e-10
    specificity = np.nanmean(np.divide(tn, tn + fp + epsilon))
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = np.mean(y_true == y_pred)
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    try:
        auroc = roc_auc_score(y_true_bin, y_pred_prob, average="weighted", multi_class="ovr")
    except ValueError:
        auroc = None
    aupr_list = []
    for i in np.unique(y_true):
        precision_vals, recall_vals, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
        aupr_list.append(auc(recall_vals, precision_vals))
    aupr = np.mean(aupr_list) if aupr_list else None
    
    return {
        "Precision_macro": precision_macro,
        "Precision_micro": precision_micro,
        "Recall_macro": recall_macro,
        "Recall_micro": recall_micro,
        "Accuracy": accuracy,
        "Specificity": specificity,
        "F1_score": f1,
        "MCC": mcc,
        "AUPR": aupr,
        "AUROC": auroc
    }

#%% =================================================
# Main Execution for All Scenarios
# =================================================
results = []
for scenario_name, scenario_info in SCENARIOS.items():
    print(f"\n=== Processing Scenario: {scenario_name} ===")
    
    if scenario_info['type'] == 'split':  # S1: Load full and split
        X_full = load_data(scenario_info['path'], scenario_info['feature_file'])
        y_full = load_separate_labels(scenario_info['path'], scenario_info['label_file'])
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_full, y_full, test_size=0.1, stratify=y_full, random_state=42
        )
        X_train, y_train = X_train_full, y_train_full
    elif scenario_info['type'] == 'separate':  # S2, S3: Separate train/test
        X_train = load_data(scenario_info['train_path'], scenario_info['feature_file'])
        y_train = load_separate_labels(scenario_info['train_path'], scenario_info['label_file'])
        X_test = load_data(scenario_info['test_path'], scenario_info['feature_file'])
        y_test = load_separate_labels(scenario_info['test_path'], scenario_info['label_file'])
    elif scenario_info['type'] == 'combined':  # Scaffold: Combined npy with label in last col
        X_train, y_train = load_combined_npy(scenario_info['train_path'], scenario_info['train_file'])
        X_test, y_test = load_combined_npy(scenario_info['test_path'], scenario_info['test_file'])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    y_categorical = to_categorical(y_encoded, num_classes=106)
    y_test_encoded = label_encoder.transform(y_test)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=106)
    
    print(f"{scenario_name} - Training data shape (X_train): {X_train.shape}")
    print(f"{scenario_name} - Training labels shape (y_train): {y_categorical.shape}")
    print(f"{scenario_name} - Test data shape (X_test): {X_test.shape}")
    print(f"{scenario_name} - Test labels shape (y_test): {y_test_categorical.shape}")
    
    # Train and validate
    models, history_by_fold, param_count, train_time, peak_vram = train_and_validate(X_train, y_categorical, scenario_name)
    
    # Final model evaluation
    final_model = models[-1]
    final_loss, final_accuracy = final_model.evaluate(X_test, y_test_categorical, verbose=0)
    print(f"{scenario_name} - Final Test Loss: {final_loss:.4f}")
    print(f"{scenario_name} - Final Test Accuracy: {final_accuracy:.4f}")
    
    # Predict and evaluate detailed metrics
    y_pred_prob = final_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test_categorical, axis=1)
    metrics = compute_detailed_metrics(y_true, y_pred, y_pred_prob, num_classes=106)
    
    # Print PR first
    print(f"\n### Precision-Recall Metrics for {scenario_name} ###")
    print(f"Precision (macro): {metrics['Precision_macro']:.4f}")
    print(f"Precision (micro): {metrics['Precision_micro']:.4f}")
    print(f"Recall (macro): {metrics['Recall_macro']:.4f}")
    print(f"Recall (micro): {metrics['Recall_micro']:.4f}")
    
    # Print other metrics
    print(f"\n### Detailed Metrics for {scenario_name} ###")
    for metric, value in {k: v for k, v in metrics.items() if k not in ['Precision_macro', 'Precision_micro', 'Recall_macro', 'Recall_micro']}.items():
        print(f"{metric}: {value:.4f}")
    
    # Store results
    scenario_metrics = {k: v for k, v in metrics.items() if k not in ['Precision_macro', 'Precision_micro', 'Recall_macro', 'Recall_micro']}
    scenario_metrics['Scenario'] = scenario_name
    scenario_metrics['Parameters'] = param_count
    scenario_metrics['Wall-Clock Time (s)'] = train_time
    scenario_metrics['Peak VRAM (GB)'] = peak_vram
    results.append(scenario_metrics)

#%% Save all results
df_results = pd.DataFrame(results)
df_results.to_csv("model_metrics_all_scenarios.csv", index=False)
print("\n=== Full Summary Across Scenarios ===")
print(df_results.round(4).to_string(index=False))

print("\n✅ Finished. Results saved to model_metrics_all_scenarios.csv")
