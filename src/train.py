import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from data_loader import SkinLesionDataLoader
from model import build_efficientnetv2_model, compile_model

# ──────────────────────────────────────────────
#  Image Quality & Font Settings
# ──────────────────────────────────────────────
DPI = 300
FONT_SIZE_MIN = 10
FONT_SIZE_MAX = 12

matplotlib.rcParams.update({
    'font.size':          FONT_SIZE_MIN,
    'axes.titlesize':     FONT_SIZE_MAX,
    'axes.labelsize':     FONT_SIZE_MAX,
    'xtick.labelsize':    FONT_SIZE_MIN,
    'ytick.labelsize':    FONT_SIZE_MIN,
    'legend.fontsize':    FONT_SIZE_MIN,
    'figure.titlesize':   FONT_SIZE_MAX,
    'figure.dpi':         DPI,
    'savefig.dpi':        DPI,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.15,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
    'font.family':        'sans-serif',
})
sns.set_style('whitegrid')


def train(data_dir, csv_path, epochs, batch_size, model_save_path, log_dir):
    """Train the EfficientNetV2 skin lesion classification model."""
    # 1. Load Data
    print("Loading data...")
    loader = SkinLesionDataLoader(data_dir=data_dir, csv_path=csv_path, batch_size=batch_size)
    train_ds, val_ds = loader.load_data()

    # 2. Build Model
    print("Building model...")
    model = build_efficientnetv2_model(num_classes=loader.num_classes)
    model = compile_model(model)

    # 3. Setup Callbacks
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        TensorBoard(log_dir=log_dir)
    ]

    # Compute Class Weights to handle imbalance
    print('Computing class weights...')
    all_labels = []
    for _, labels in train_ds:
        all_labels.extend(labels.numpy())
    
    classes = np.unique(all_labels)
    weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    class_weight_dict = {int(c): w for c, w in zip(classes, weights)}
    print('Class Weights:', class_weight_dict)

    # 4. Train Model
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    print("Training complete. Best model saved to:", model_save_path)
    return history, model, val_ds, loader


def save_training_history(history, results_dir):
    """Plot and save training accuracy & loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax = axes[0]
    ax.plot(history.history['accuracy'],    linewidth=2, marker='o', markersize=4, label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], linewidth=2, marker='s', markersize=4, label='Val Accuracy')
    ax.set_title('Model Accuracy', fontsize=FONT_SIZE_MAX, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=FONT_SIZE_MAX)
    ax.set_xlabel('Epoch', fontsize=FONT_SIZE_MAX)
    ax.legend(loc='lower right', fontsize=FONT_SIZE_MIN)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_MIN)
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1]
    ax.plot(history.history['loss'],    linewidth=2, marker='o', markersize=4, label='Train Loss')
    ax.plot(history.history['val_loss'], linewidth=2, marker='s', markersize=4, label='Val Loss')
    ax.set_title('Model Loss', fontsize=FONT_SIZE_MAX, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=FONT_SIZE_MAX)
    ax.set_xlabel('Epoch', fontsize=FONT_SIZE_MAX)
    ax.legend(loc='upper right', fontsize=FONT_SIZE_MIN)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_MIN)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    history_path = os.path.join(results_dir, 'training_history.png')
    fig.savefig(history_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved training history plot to {history_path}")


def save_confusion_matrix(y_true, y_pred, class_names, results_dir):
    """Plot and save the confusion matrix as a high-quality image."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'size': FONT_SIZE_MIN, 'fontweight': 'bold'},
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    ax.set_title('Confusion Matrix', fontsize=FONT_SIZE_MAX, fontweight='bold', pad=15)
    ax.set_ylabel('True Label', fontsize=FONT_SIZE_MAX)
    ax.set_xlabel('Predicted Label', fontsize=FONT_SIZE_MAX)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_MIN)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()

    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")


def save_classification_report_csv(y_true, y_pred, class_names, results_dir):
    """Save the classification report as a well-structured CSV file."""
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )

    # Per-class rows
    rows = []
    for cls_name in class_names:
        metrics = report_dict[cls_name]
        rows.append({
            'Class':     cls_name,
            'Precision': round(metrics['precision'], 4),
            'Recall':    round(metrics['recall'],    4),
            'F1-Score':  round(metrics['f1-score'],  4),
            'Support':   int(metrics['support'])
        })

    # Blank separator row
    rows.append({
        'Class': '', 'Precision': '', 'Recall': '', 'F1-Score': '', 'Support': ''
    })

    # Accuracy row
    accuracy_val = report_dict['accuracy']
    total_support = int(report_dict['weighted avg']['support'])
    rows.append({
        'Class':     'accuracy',
        'Precision': '',
        'Recall':    '',
        'F1-Score':  round(accuracy_val, 4),
        'Support':   total_support
    })

    # Macro / weighted average rows
    for avg_key in ['macro avg', 'weighted avg']:
        m = report_dict[avg_key]
        rows.append({
            'Class':     avg_key,
            'Precision': round(m['precision'], 4),
            'Recall':    round(m['recall'],    4),
            'F1-Score':  round(m['f1-score'],  4),
            'Support':   int(m['support'])
        })

    report_df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, 'classification_report.csv')
    report_df.to_csv(csv_path, index=False)
    print(f"Saved classification report CSV to {csv_path}")
    print(report_df.to_string(index=False))
    return report_df


def save_classification_report_image(report_df, class_names, results_dir):
    """Render the classification report DataFrame as a styled table image."""
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(report_df) + 1.5)))
    ax.axis('off')
    ax.set_title('Classification Report', fontsize=FONT_SIZE_MAX, fontweight='bold', pad=20)

    table = ax.table(
        cellText=report_df.values,
        colLabels=report_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE_MIN)
    table.scale(1.2, 1.8)

    # Header styling
    for col_idx in range(len(report_df.columns)):
        cell = table[0, col_idx]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=FONT_SIZE_MAX)
        cell.set_edgecolor('white')

    # Row styling
    num_class_rows = len(class_names)
    for row_idx in range(1, len(report_df) + 1):
        for col_idx in range(len(report_df.columns)):
            cell = table[row_idx, col_idx]
            cell.set_edgecolor('#bdc3c7')

            if row_idx <= num_class_rows:
                cell.set_facecolor('#ecf0f1' if row_idx % 2 == 0 else '#ffffff')
            elif row_idx == num_class_rows + 1:
                cell.set_facecolor('#f8f9fa')
                cell.set_edgecolor('#f8f9fa')
            else:
                cell.set_facecolor('#d5e8d4')
                cell.set_text_props(fontweight='bold')

    fig.tight_layout()

    img_path = os.path.join(results_dir, 'classification_report.png')
    fig.savefig(img_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved classification report image to {img_path}")


def save_per_class_performance(report_df, class_names, results_dir):
    """Save a grouped bar chart of per-class precision / recall / F1."""
    class_df = report_df[report_df['Class'].isin(class_names)].copy()
    class_df = class_df.set_index('Class')
    metric_cols = ['Precision', 'Recall', 'F1-Score']
    class_df[metric_cols] = class_df[metric_cols].astype(float)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    bar_width = 0.25
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for i, (metric, color) in enumerate(zip(metric_cols, colors)):
        bars = ax.bar(
            x + i * bar_width, class_df[metric], bar_width,
            label=metric, color=color, edgecolor='white', linewidth=0.5
        )
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontsize=FONT_SIZE_MIN - 2, fontweight='bold'
                )

    ax.set_title('Per-Class Performance Metrics', fontsize=FONT_SIZE_MAX, fontweight='bold')
    ax.set_ylabel('Score', fontsize=FONT_SIZE_MAX)
    ax.set_xlabel('Class', fontsize=FONT_SIZE_MAX)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=FONT_SIZE_MIN)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_MIN)
    ax.legend(fontsize=FONT_SIZE_MIN, loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()

    bar_path = os.path.join(results_dir, 'per_class_performance.png')
    fig.savefig(bar_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved per-class performance chart to {bar_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Skin Lesion Detection Model')
    parser.add_argument('--data_dir',    type=str, default='../data/images',              help='Path to images directory')
    parser.add_argument('--csv_path',    type=str, default='../data/HAM10000_metadata.csv', help='Path to metadata CSV')
    parser.add_argument('--results_dir', type=str, default='../data/results',              help='Path to save results')
    parser.add_argument('--model_save',  type=str, default='../models/best_model.h5',      help='Path to save the best model')
    parser.add_argument('--log_dir',     type=str, default='./logs',                       help='Path for TensorBoard logs')
    parser.add_argument('--epochs',      type=int, default=20,                             help='Number of epochs to train')
    parser.add_argument('--batch_size',  type=int, default=32,                             help='Batch size for training')
    
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_save), exist_ok=True)
    
    # Run Training
    history, model, val_ds, loader = train(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_save,
        log_dir=args.log_dir
    )

    # 1. Save Training History Plot
    save_training_history(history, args.results_dir)

    # 2. Evaluate on validation set
    print("Evaluating on validation set...")
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 3. Save Confusion Matrix
    save_confusion_matrix(y_true, y_pred, loader.classes, args.results_dir)

    # 4. Save Classification Report (CSV + Image)
    report_df = save_classification_report_csv(y_true, y_pred, loader.classes, args.results_dir)
    save_classification_report_image(report_df, loader.classes, args.results_dir)

    # 5. Save Per-Class Performance Bar Chart
    save_per_class_performance(report_df, loader.classes, args.results_dir)

    print(f"\nAll results saved to: {args.results_dir}")
