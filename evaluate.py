import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from dataset import RadarGestureDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

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
        weights_file = "best_lstm_model.pth"
        output_image = "confusion_matrix_lstm.png"
        model_name = "LSTM"
    elif choice == '2':
        print("Initializing SRDST Adapted Model architecture...")
        model = SRDST_Adapted_Network(num_classes=12).to(device)
        weights_file = "best_srdst_model.pth"
        output_image = "confusion_matrix_srdst.png"
        model_name = "SRDST"
    else:
        if choice != '1':
            print("Invalid input. Defaulting to FMCW Lightweight Model...")
        print("Initializing FMCW Lightweight Model architecture...")
        model = GestureRecognitionNetwork(num_classes=12).to(device)
        weights_file = "best_fmcw_model.pth"
        output_image = "confusion_matrix_fmcw.png"
        model_name = "FMCW TCN"

    # 3. Recreate the Dataset and exact same Test Split
    full_dataset = RadarGestureDataset(data_dir="Data", seq_length=40)
    labels = [sample[1] for sample in full_dataset.samples]
    classes = full_dataset.classes 
    
    _, temp_idx = train_test_split(range(len(full_dataset)), test_size=0.4, stratify=labels, random_state=42)
    temp_labels = [labels[i] for i in temp_idx]
    _, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=42)
    
    test_dataset = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Test set loaded with {len(test_dataset)} samples.")

    # 4. Load the Trained Model Weights
    try:
        model.load_state_dict(torch.load(weights_file))
        print(f"Successfully loaded '{weights_file}'")
    except FileNotFoundError:
        print(f"Error: '{weights_file}' not found. Please run main.py to train the {model_name} model first.")
        return

    # 5. Evaluation Loop
    model.eval()
    all_preds = []
    all_trues = []
    
    print(f"Running inference on test set using {model_name}...")
    with torch.no_grad():
        for range_seq, vel_seq, az_seq, batch_labels in test_loader:
            range_seq, vel_seq, az_seq = range_seq.to(device), vel_seq.to(device), az_seq.to(device)
            
            outputs = model(range_seq, vel_seq, az_seq)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_trues.extend(batch_labels.cpu().numpy())

    # 6. Calculate Metrics
    acc = np.mean(np.array(all_preds) == np.array(all_trues))
    print(f"\n--- Final Test Accuracy ({model_name}): {acc * 100:.2f}% ---")
    
    print("\nClassification Report:")
    print(classification_report(all_trues, all_preds, target_names=classes))

    # 7. Plot the Confusion Matrix
    cm = confusion_matrix(all_trues, all_preds)
    
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f"{model_name} Gesture Recognition Confusion Matrix")
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