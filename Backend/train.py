import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from Backend.config import EPOCHS, OUTPUT_DIR
from Backend.data_preprocessing import get_data_generators
from Backend.model import build_model

def plot_history(history):
    """
    Plots and saves training & validation accuracy and loss.
    """
    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_plot.png'))
    plt.close()

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_plot.png'))
    plt.close()
    
    print(f"Plots saved to {OUTPUT_DIR}")

def train():
    """
    Executes the model training pipeline.
    """
    print("Loading data generators...")
    train_generator, val_generator, _ = get_data_generators()

    print("Building model...")
    model = build_model()
    
    # Callbacks
    model_save_path = os.path.join(OUTPUT_DIR, 'brain_tumor_model.keras')
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        model_save_path, 
        monitor='val_accuracy', 
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=5, 
        min_lr=1e-6,
        verbose=1
    )

    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )

    print("Training complete. Plotting history...")
    plot_history(history)
    
    print(f"Model saved to: {model_save_path}")

if __name__ == '__main__':
    train()
