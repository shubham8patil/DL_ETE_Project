import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from Backend.config import TEST_DIR, OUTPUT_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
from Backend.data_preprocessing import get_data_generators

def evaluate_model():
    """
    Evaluates the saved model on the test set, prints metrics, and saves plots.
    """
    model_path = os.path.join(OUTPUT_DIR, 'brain_tumor_model.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the model first.")
        return

    print("Loading model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Standard load failed, attempting weight-only fallback... ({e})")
        from Backend.model import build_model
        model = build_model()
        model.load_weights(model_path)

    print("Loading test data...")
    _, _, test_generator = get_data_generators()
    
    print("Evaluating...")
    # Evaluate returns [loss, accuracy]
    results = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]*100:.2f}%")

    print("Generating predictions...")
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    class_labels = list(test_generator.class_indices.keys())

    # 1. Classification Report
    print("\\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)
    
    # Save report to text file
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    print(f"Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")

    # 3. Sample Predictions Plot
    plot_sample_predictions(model, test_generator, class_labels)

def plot_sample_predictions(model, test_generator, class_labels):
    """
    Plots a grid of sample test images with their true and predicted labels.
    """
    # Get a batch of test data
    test_generator.reset()
    images, labels = next(test_generator)
    preds = model.predict(images)
    
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels, axis=1)

    plt.figure(figsize=(15, 10))
    # Plot up to 15 images
    num_to_plot = min(15, len(images))
    for i in range(num_to_plot):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[i])
        
        true_label = class_labels[true_classes[i]]
        pred_label = class_labels[pred_classes[i]]
        
        # Color green if correct, red if wrong
        color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f"T: {true_label}\\nP: {pred_label}", color=color, fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_predictions.png'))
    plt.close()
    print(f"Sample predictions saved to {OUTPUT_DIR}/sample_predictions.png")

if __name__ == '__main__':
    evaluate_model()
