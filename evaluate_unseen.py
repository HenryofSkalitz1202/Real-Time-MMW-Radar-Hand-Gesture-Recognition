import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from dataset import RadarGestureDataset
from torch.utils.data import DataLoader

from model.one_d_tcn import GestureRecognitionNetwork
from model.srdst import SRDST_Adapted_Network 
from model.lstm import LSTM_Gesture_Network

def evaluate_test_set():
    print("="*50)
    print("Model Evaluation")
    print("="*50)
    
    # 1. Model Selection UI
    print("Available Models for Evaluation:")
    print("1: FMCW Lightweight (DS-TCN)")
    print("2: SRDST Adapted (Dual-Stream Transformer)")
    print("3: LSTM (Grobelny & Narbudowicz)")
    
    choice = input("Enter the number of the model to evaluate (1, 2, or 3): ").strip()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating on device: {device}")

    # 2. Initialize the correct architecture and filenames based on choice
    if choice == '3':
        print("Initializing LSTM Model architecture...")
        model = LSTM_Gesture_Network(num_classes=12).to(device)
        weights_file = "weights/best_lstm_model.pth"
        output_image = "confusion_matrix_lstm.png"
        model_name = "LSTM"
    elif choice == '2':
        print("Initializing SRDST Adapted Model architecture...")
        model = SRDST_Adapted_Network(num_classes=12).to(device)
        weights_file = "weights/best_srdst_model.pth"
        output_image = "confusion_matrix_srdst.png"
        model_name = "SRDST"
    else:
        if choice != '1':
            print("Invalid input. Defaulting to FMCW Lightweight Model...")
        print("Initializing FMCW Lightweight Model architecture...")
        model = GestureRecognitionNetwork(num_classes=12).to(device)
        weights_file = "weights/best_fmcw_model.pth"
        output_image = "confusion_matrix_fmcw.png"
        model_name = "FMCW TCN"

    # 3. Load the completely Unseen Test Dataset (100% of target_folder)
    print("Scanning directory and building dataset...")
    test_dataset = RadarGestureDataset(data_dir="target_folder", seq_length=40)
    classes = test_dataset.classes  # These are your 6 new folder names
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Test set loaded with {len(test_dataset)} samples.")

    # 4. Load the Trained Model Weights
    try:
        model.load_state_dict(torch.load(weights_file, map_location=device))
        print(f"Successfully loaded '{weights_file}'")
    except FileNotFoundError:
        print(f"Error: '{weights_file}' not found. Please train the {model_name} model first.")
        return

    # 5. Evaluation Loop
    model.eval()
    all_preds = []
    all_trues = []
    
    print(f"Running inference on test set using {model_name}...")
    with torch.no_grad():
        for range_seq, vel_seq, az_seq, batch_labels in test_loader:
            range_seq = range_seq.to(device)
            vel_seq = vel_seq.to(device)
            az_seq = az_seq.to(device)
            
            outputs = model(range_seq, vel_seq, az_seq)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_trues.extend(batch_labels.cpu().numpy())

    # ---------------------------------------------------------
    # 6. Label Alignment & Metrics
    # ---------------------------------------------------------
    # The original 12 classes the model was trained on
    original_classes = ["Circle CCW", "Circle CW", "Down", "Expand", "Pull", 
                        "Push", "Shrink", "Swipe Left", "Swipe Right", "Tap", "Up", "Wave"]
    
    # Map your new test folder names to the model's original concepts
    name_translation = {
        "hand_away": "Push",
        "hand_closer": "Pull",
        "hand_down": "Down",
        "hand_up": "Up",
        "hand_to_left": "Swipe Left",
        "hand_to_right": "Swipe Right"
    }

    # Translate the dataset indices to match the model's indices
    aligned_trues = []
    for true_idx in all_trues:
        folder_name = classes[true_idx]              # e.g., 'hand_away'
        old_name = name_translation[folder_name]     # e.g., 'Push'
        model_idx = original_classes.index(old_name) # e.g., 5
        aligned_trues.append(model_idx)

    # Calculate actual accuracy
    acc = np.mean(np.array(all_preds) == np.array(aligned_trues))
    print(f"\n--- Final Test Accuracy ({model_name}): {acc * 100:.2f}% ---")
    
    # Get only the indices and names for the 6 gestures we actually tested
    tested_indices = [original_classes.index(name_translation[c]) for c in classes]
    tested_names = [name_translation[c] for c in classes]

    print("\nClassification Report:")
    print(classification_report(aligned_trues, all_preds, labels=tested_indices, target_names=tested_names))

    # ---------------------------------------------------------
    # 7. Plot the Confusion Matrix
    # ---------------------------------------------------------
    cm = confusion_matrix(aligned_trues, all_preds, labels=tested_indices)
    
    plt.figure(figsize=(10, 8))
    # Add a small epsilon (1e-9) to prevent divide-by-zero errors
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) 
    
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=tested_names, yticklabels=tested_names)
    
    plt.title(f"{model_name} Unseen Evaluation Matrix")
    plt.ylabel("True Gesture")
    plt.xlabel("Predicted Gesture")
    plt.xticks(rotation=45, ha='right') 
    plt.tight_layout()
    
    # Save dynamically based on the chosen model
    plt.savefig(output_image, dpi=300)
    print(f"\nConfusion matrix saved as '{output_image}'.")
    plt.show()

if __name__ == "__main__":
    evaluate_test_set()