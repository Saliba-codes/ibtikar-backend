#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IMPROVED Fine-tuning with:
1. Automatic outlier detection and removal
2. More aggressive class balancing
3. Better hyperparameters for imbalanced data
4. Multi-threshold evaluation
5. Confidence-calibrated predictions
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report, roc_auc_score
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import Counter

# ----------------------------- Logging ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("improved_finetune")

# ----------------------------- Dataset ---------------------------------
class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts = df["input_text"].tolist()
        self.labels = df["Label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
        )
        item = {k: v for k, v in enc.items()}
        item["labels"] = self.labels[idx]
        return item

# ----------------------------- Data Cleaning ---------------------------
def detect_outliers(df, text_col='text', label_col='Label'):
    """Detect potential mislabeled examples based on keywords"""
    
    HATE_KEYWORDS = [
        'خنزير', 'خنازير', 'كلب', 'كلاب', 'حمار', 'بقر', 'قرد',
        'ديوث', 'عاهر', 'حقير', 'نجس', 'ارهابي', 'خائن', 'مجرم',
        'وسخ', 'قذر', 'حثالة', 'بلطجي'
    ]
    
    suspicious_indices = []
    
    for idx, row in df.iterrows():
        if row[label_col] == 0:  # Check Class 0 for hate keywords
            text = str(row[text_col]).lower()
            hate_count = sum(1 for word in HATE_KEYWORDS if word in text)
            
            if hate_count >= 2:  # Multiple hate words but labeled as normal
                suspicious_indices.append(idx)
    
    return suspicious_indices

# ----------------------------- Custom Trainer --------------------------
class ImbalancedTrainer(Trainer):
    """Enhanced trainer for heavily imbalanced datasets"""
    
    def __init__(self, class_weights=None, sampler_weights=None, 
                 focal_loss=False, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        self.sampler_weights = sampler_weights
        self.focal_loss = focal_loss
        self.focal_gamma = focal_gamma
        
        if focal_loss:
            logger.info(f"Using Focal Loss with gamma={focal_gamma}")

    def get_train_dataloader(self):
        """Use weighted sampler to oversample minority class"""
        if self.train_dataset is None or self.sampler_weights is None:
            return super().get_train_dataloader()
        
        from torch.utils.data import DataLoader
        sampler = WeightedRandomSampler(
            weights=self.sampler_weights,
            num_samples=len(self.sampler_weights),
            replacement=True
        )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.focal_loss:
            # Focal loss for hard examples
            ce_loss = torch.nn.functional.cross_entropy(
                logits, labels, reduction='none',
                weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
            )
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
            return (focal_loss, outputs) if return_outputs else focal_loss
        else:
            # Standard weighted cross entropy
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
                label_smoothing=self.args.label_smoothing_factor
            )
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

# ----------------------------- Metrics ---------------------------------
def compute_metrics_fn():
    def _fn(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        
        # Calculate AUC if possible
        try:
            auc = roc_auc_score(labels, probs[:, 1])
        except:
            auc = 0.0
        
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_class1": f1_score(labels, preds, labels=[1], average="binary", zero_division=0),
            "recall_class1": recall_score(labels, preds, labels=[1], average="binary", zero_division=0),
            "precision_class1": precision_score(labels, preds, labels=[1], average="binary", zero_division=0),
            "auc": auc,
        }
    return _fn

# ----------------------- Multi-Threshold Evaluation --------------------
def evaluate_multiple_thresholds(y_true, probs_pos):
    """Find best thresholds for different metrics"""
    
    thresholds = np.linspace(0.05, 0.95, 37)
    results = []
    
    for t in thresholds:
        y_pred = (probs_pos >= t).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        
        results.append({
            'threshold': t,
            'accuracy': acc,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'f1_x_recall': f1 * recall,  # Combined metric
        })
    
    df = pd.DataFrame(results)
    
    # Find best thresholds for different objectives
    best_f1_idx = df['f1'].idxmax()
    best_recall_idx = df['recall'].idxmax()
    best_balanced_idx = df['f1_x_recall'].idxmax()
    
    return {
        'best_f1': {
            'threshold': df.loc[best_f1_idx, 'threshold'],
            'f1': df.loc[best_f1_idx, 'f1'],
            'recall': df.loc[best_f1_idx, 'recall'],
            'precision': df.loc[best_f1_idx, 'precision'],
        },
        'best_recall': {
            'threshold': df.loc[best_recall_idx, 'threshold'],
            'f1': df.loc[best_recall_idx, 'f1'],
            'recall': df.loc[best_recall_idx, 'recall'],
            'precision': df.loc[best_recall_idx, 'precision'],
        },
        'best_balanced': {
            'threshold': df.loc[best_balanced_idx, 'threshold'],
            'f1': df.loc[best_balanced_idx, 'f1'],
            'recall': df.loc[best_balanced_idx, 'recall'],
            'precision': df.loc[best_balanced_idx, 'precision'],
        },
        'all_results': df
    }

# ----------------------------- Main ------------------------------------
def main():
    # ============ IMPROVED CONFIGURATION ============
    MODEL_CHECKPOINT = "out_marbv2"
    CSV_FILE = "Clean_Normalized.csv"
    TEXT_COLUMN = "text"
    LABEL_COLUMN = "Label"
    OUTPUT_DIR = "out_marbv2_improved"
    
    # More aggressive settings for imbalanced data
    MAX_LENGTH = 256
    BATCH_SIZE = 8  # Smaller batch for stability
    EVAL_BATCH_SIZE = 32
    GRAD_ACCUM = 4  # More accumulation
    LEARNING_RATE = 3e-6  # Lower LR for stability
    NUM_EPOCHS = 8  # More epochs
    WEIGHT_DECAY = 0.01  # Less weight decay
    LABEL_SMOOTHING = 0.0  # No smoothing (can hurt minority class)
    WARMUP_RATIO = 0.15  # More warmup
    
    # Sampling strategy
    MINORITY_WEIGHT = 5.0  # Even more aggressive oversampling
    
    # Loss strategy
    USE_FOCAL_LOSS = True  # Try focal loss for hard examples
    FOCAL_GAMMA = 2.0
    
    # Data cleaning
    REMOVE_OUTLIERS = True  # Auto-remove suspicious mislabeled examples
    
    SEED = 42
    # ================================================
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    logger.info("="*80)
    logger.info("IMPROVED FINE-TUNING FOR IMBALANCED DATA")
    logger.info("="*80)
    logger.info(f"Model: {MODEL_CHECKPOINT}")
    logger.info(f"Minority weight: {MINORITY_WEIGHT}x")
    logger.info(f"Focal loss: {USE_FOCAL_LOSS}")
    logger.info(f"Remove outliers: {REMOVE_OUTLIERS}")
    
    # Load data
    logger.info(f"\nLoading: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    logger.info(f"Initial samples: {len(df)}")
    
    # Clean data
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df = df.drop_duplicates(subset=[TEXT_COLUMN])
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN].str.len() >= 3]
    
    # Detect and remove outliers
    if REMOVE_OUTLIERS:
        logger.info("\nDetecting potential mislabeled examples...")
        outlier_indices = detect_outliers(df, TEXT_COLUMN, LABEL_COLUMN)
        
        if len(outlier_indices) > 0:
            logger.info(f"Found {len(outlier_indices)} suspicious Class 0 examples")
            logger.info(f"Removing {min(len(outlier_indices), 500)} most suspicious...")
            
            # Remove up to 500 most suspicious
            df = df.drop(outlier_indices[:500])
            logger.info(f"Samples after cleaning: {len(df)}")
    
    # Check class distribution
    class_dist = df[LABEL_COLUMN].value_counts()
    logger.info(f"\nClass distribution:")
    for label, count in class_dist.items():
        logger.info(f"  Class {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Map labels
    unique_labels = sorted(df[LABEL_COLUMN].unique())
    if len(unique_labels) != 2:
        raise ValueError(f"Expected 2 classes, got {len(unique_labels)}")
    
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label = {i: str(lbl) for lbl, i in label2id.items()}
    df["Label_id"] = df[LABEL_COLUMN].map(label2id)
    df["input_text"] = df[TEXT_COLUMN]
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_df, temp_df = train_test_split(
        df[["input_text", "Label_id"]], 
        test_size=0.3,
        random_state=SEED,
        stratify=df["Label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df["Label_id"]
    )
    
    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_df)}")
    logger.info(f"  Val:   {len(val_df)}")
    logger.info(f"  Test:  {len(test_df)}")
    
    # Calculate class weights on training data
    train_labels = train_df["Label_id"].values
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
    
    logger.info(f"\nClass weights (from train):")
    logger.info(f"  Class 0: {class_weights[0]:.4f}")
    logger.info(f"  Class 1: {class_weights[1]:.4f}")
    logger.info(f"  Ratio: {class_weights[1]/class_weights[0]:.2f}x")
    
    # Create sampler weights for oversampling
    sampler_weights = np.where(train_labels == 1, MINORITY_WEIGHT, 1.0).astype(np.float32)
    logger.info(f"Sampler: Class 1 will be sampled {MINORITY_WEIGHT}x more")
    
    # Load model and tokenizer
    logger.info(f"\nLoading model from {MODEL_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    
    # Create datasets
    train_ds = TextDataset(train_df, tokenizer, MAX_LENGTH)
    val_ds = TextDataset(val_df, tokenizer, MAX_LENGTH)
    test_ds = TextDataset(test_df, tokenizer, MAX_LENGTH)
    
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="recall_class1",  # Optimize for Class 1 recall
        greater_is_better=True,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        label_smoothing_factor=LABEL_SMOOTHING,
        save_total_limit=3,
        report_to="none",
        seed=SEED,
    )
    
    # Create trainer
    trainer = ImbalancedTrainer(
        class_weights=class_weights,
        sampler_weights=sampler_weights,
        focal_loss=USE_FOCAL_LOSS,
        focal_gamma=FOCAL_GAMMA,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING...")
    logger.info("="*80 + "\n")
    
    # Train
    trainer.train()
    
    # Multi-threshold evaluation
    logger.info("\n" + "="*80)
    logger.info("FINDING OPTIMAL THRESHOLDS")
    logger.info("="*80)
    
    val_preds = trainer.predict(val_ds)
    val_logits = val_preds.predictions
    val_labels = val_preds.label_ids
    val_probs_pos = torch.softmax(torch.tensor(val_logits), dim=1).numpy()[:, 1]
    
    threshold_results = evaluate_multiple_thresholds(val_labels, val_probs_pos)
    
    logger.info("\nBest thresholds for different objectives:")
    logger.info(f"\n1. BEST F1:")
    logger.info(f"   Threshold: {threshold_results['best_f1']['threshold']:.4f}")
    logger.info(f"   F1: {threshold_results['best_f1']['f1']:.4f}")
    logger.info(f"   Recall: {threshold_results['best_f1']['recall']:.4f}")
    logger.info(f"   Precision: {threshold_results['best_f1']['precision']:.4f}")
    
    logger.info(f"\n2. BEST RECALL (catches more Class 1):")
    logger.info(f"   Threshold: {threshold_results['best_recall']['threshold']:.4f}")
    logger.info(f"   F1: {threshold_results['best_recall']['f1']:.4f}")
    logger.info(f"   Recall: {threshold_results['best_recall']['recall']:.4f}")
    logger.info(f"   Precision: {threshold_results['best_recall']['precision']:.4f}")
    
    logger.info(f"\n3. BALANCED (F1 × Recall):")
    logger.info(f"   Threshold: {threshold_results['best_balanced']['threshold']:.4f}")
    logger.info(f"   F1: {threshold_results['best_balanced']['f1']:.4f}")
    logger.info(f"   Recall: {threshold_results['best_balanced']['recall']:.4f}")
    logger.info(f"   Precision: {threshold_results['best_balanced']['precision']:.4f}")
    
    # Use balanced threshold
    best_threshold = threshold_results['best_balanced']['threshold']
    
    # Test set evaluation
    logger.info("\n" + "="*80)
    logger.info(f"TEST SET EVALUATION (threshold={best_threshold:.4f})")
    logger.info("="*80)
    
    test_preds = trainer.predict(test_ds)
    test_logits = test_preds.predictions
    test_labels = test_preds.label_ids
    test_probs_pos = torch.softmax(torch.tensor(test_logits), dim=1).numpy()[:, 1]
    test_pred_ids = (test_probs_pos >= best_threshold).astype(int)
    
    # Classification report
    report = classification_report(
        test_labels, test_pred_ids,
        target_names=[id2label[0], id2label[1]],
        digits=4
    )
    logger.info(f"\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_pred_ids)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"        pred_0  pred_1")
    logger.info(f"true_0  {cm[0][0]:6d}  {cm[0][1]:6d}")
    logger.info(f"true_1  {cm[1][0]:6d}  {cm[1][1]:6d}")
    
    class_1_recall = cm[1][1] / cm[1].sum() if cm[1].sum() > 0 else 0
    class_1_precision = cm[1][1] / (cm[0][1] + cm[1][1]) if (cm[0][1] + cm[1][1]) > 0 else 0
    
    logger.info(f"\nClass 1 Performance:")
    logger.info(f"  Recall: {class_1_recall:.4f}")
    logger.info(f"  Precision: {class_1_precision:.4f}")
    
    # Save everything
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    with open(os.path.join(OUTPUT_DIR, "thresholds.json"), "w") as f:
        json.dump({
            'best_f1_threshold': float(threshold_results['best_f1']['threshold']),
            'best_recall_threshold': float(threshold_results['best_recall']['threshold']),
            'recommended_threshold': float(best_threshold),
        }, f, indent=2)
    
    threshold_results['all_results'].to_csv(
        os.path.join(OUTPUT_DIR, "threshold_analysis.csv"),
        index=False
    )
    
    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING COMPLETE")
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info(f"Recommended threshold: {best_threshold:.4f}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()