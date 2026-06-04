import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

from dataset import RadarGestureDataset
from model.one_d_tcn import GestureRecognitionNetwork 
from model.srdst import SRDST_Adapted_Network
from model.lstm import LSTM_Gesture_Network

def main():
    print("="*50)
    print("mmWave Hand Gesture Recognition Pipeline (4-Feature Edition)")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Dataset FIRST to dynamically get num_classes ---
    print("\nLoading dataset...")
    full_dataset = RadarGestureDataset(data_dir="Data_51cm", seq_length=40)
    labels = [sample[1] for sample in full_dataset.samples]
    
    NUM_CLASSES = len(full_dataset.classes)
    print(f"Detected {NUM_CLASSES} classes: {full_dataset.classes}")

    REST_INDEX = full_dataset.class_to_idx.get("rest", -1)
    if REST_INDEX != -1:
        print(f"Found 'Rest' class at index {REST_INDEX}. Will exclude from noise augmentation.")
    else:
        print("Warning: No 'Rest' class found. All classes will receive noise augmentation.")

    # --- 2. Model Selection UI ---
    print("\nAvailable Models:")
    print("1: FMCW Lightweight (DS-TCN + ECA)         [UPDATED FOR 4 INPUTS]")
    print("2: SRDST Adapted (Dual-Stream Transformer) [UPDATED FOR 4 INPUTS]")
    print("3: LSTM (Grobelny & Narbudowicz)           [UPDATED FOR 4 INPUTS]")
    
    choice = input("Enter the number of the model to train (1, 2, or 3): ").strip()

    if choice == '3':
        print(f"Initializing LSTM Model for {NUM_CLASSES} classes...")
        model = LSTM_Gesture_Network(num_classes=NUM_CLASSES).to(device)
        save_filename = "best_lstm_model_v5.pth"
        model_name = "LSTM"
    elif choice == '2':
        print(f"Initializing SRDST Adapted Model for {NUM_CLASSES} classes...")
        model = SRDST_Adapted_Network(num_classes=NUM_CLASSES).to(device)
        save_filename = "best_srdst_model_v5.pth"
        model_name = "SRDST"
    else:
        if choice != '1':
            print("Invalid input. Defaulting to FMCW Lightweight Model...")
        print(f"Initializing FMCW Lightweight Model for {NUM_CLASSES} classes...")
        model = GestureRecognitionNetwork(num_classes=NUM_CLASSES).to(device)
        save_filename = "best_fmcw_model_v8.51.pth"
        model_name = "TCN"

    # --- 3. Hyperparameters ---
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 16

    # --- 4. Split the Dataset ---
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    print(f"Data Split - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 5. Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # --- 6. Training Loop ---
    best_val_acc = 0.0
    print(f"\nStarting Training for {model_name}...")

    for epoch in range(num_epochs):
        # -- Train Phase --
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train", leave=False)
        for range_seq, vel_seq, az_seq, el_seq, batch_labels in loop:
            range_seq = range_seq.to(device)
            vel_seq = vel_seq.to(device)
            az_seq = az_seq.to(device)
            el_seq = el_seq.to(device)
            batch_labels = batch_labels.to(device)

            # --- DATA AUGMENTATION (Training Only) ---
            # 1. Amplitude Scaling (Simulates distance variations)
            scale = torch.empty(batch_labels.size(0), 1, 1, device=device).uniform_(0.8, 1.2)
            range_seq = range_seq * scale
            vel_seq = vel_seq * scale
            az_seq = az_seq * scale
            el_seq = el_seq * scale

            # 2. ASYMMETRIC Gaussian Noise
            # R/V can handle 0.02, but Az/El need 0.005 to protect the Left/Right phase boundary
            noise_level_rv = 0.02
            noise_level_azel = 0.005
            
            range_seq = range_seq + (torch.randn_like(range_seq) * noise_level_rv)
            vel_seq = vel_seq + (torch.randn_like(vel_seq) * noise_level_rv)
            az_seq = az_seq + (torch.randn_like(az_seq) * noise_level_azel)
            el_seq = el_seq + (torch.randn_like(el_seq) * noise_level_azel)

            # 3. Temporal Shifting (Crucial for Time-Series)
            shift = torch.randint(-4, 5, (1,)).item()
            if shift != 0:
                range_seq = torch.roll(range_seq, shifts=shift, dims=2)
                vel_seq = torch.roll(vel_seq, shifts=shift, dims=2)
                az_seq = torch.roll(az_seq, shifts=shift, dims=2)
                el_seq = torch.roll(el_seq, shifts=shift, dims=2)
            # -----------------------------------------

            optimizer.zero_grad()
            
            # ---> UNIFIED FORWARD PASS: All 3 models now natively accept 4 inputs!
            outputs = model(range_seq, vel_seq, az_seq, el_seq)
                
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
            
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = (correct_train / total_train) * 100

        #scheduler.step()

        # -- Validation Phase --
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for range_seq, vel_seq, az_seq, el_seq, batch_labels in val_loader:
                range_seq = range_seq.to(device)
                vel_seq = vel_seq.to(device)
                az_seq = az_seq.to(device)
                el_seq = el_seq.to(device)
                batch_labels = batch_labels.to(device)
                
                # ---> UNIFIED FORWARD PASS
                outputs = model(range_seq, vel_seq, az_seq, el_seq)
                    
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_labels.size(0)
                correct_val += (predicted == batch_labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct_val / total_val) * 100
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_filename)
            print(f"  -> Saved new best model to {save_filename}!")

    print(f"\nTraining Finished! Best Validation Accuracy for {model_name}: {best_val_acc:.2f}%")

    # ==========================================
    # DEBUGGING: Post-Training Confusion Matrix
    # ==========================================
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    print(f"\n--- Generating Debug Confusion Matrix for {model_name} ---")
    model.load_state_dict(torch.load(save_filename))
    model.eval()
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for range_seq, vel_seq, az_seq, el_seq, batch_labels in val_loader:
            range_seq = range_seq.to(device)
            vel_seq = vel_seq.to(device)
            az_seq = az_seq.to(device)
            el_seq = el_seq.to(device)
            
            # ---> UNIFIED FORWARD PASS
            outputs = model(range_seq, vel_seq, az_seq, el_seq)
                
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_trues.extend(batch_labels.cpu().numpy())
            
    cm = confusion_matrix(all_trues, all_preds)
    plt.figure(figsize=(10, 8))
    
    # Normalize by row to get percentages
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
    plt.title(f"Validation Confusion Matrix ({model_name})")
    plt.ylabel("True Gesture")
    plt.xlabel("Predicted Gesture")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Dynamic filename ensures no overwrites!
    cm_filename = f"debug_confusion_matrix_51_{model_name}.png"
    plt.savefig(cm_filename, dpi=300)
    print(f"Saved '{cm_filename}'. Please review it!")

    # ==========================================
    # DEBUGGING: Misclassified File Logger
    # ==========================================
    print("\n--- Diagnosing Misclassified Files ---")
    
    # Ensure the model is in eval mode and best weights are loaded
    model.load_state_dict(torch.load(save_filename))
    model.eval()

    log_filename = f"misclassified_log_51_{model_name}.csv"
    
    with open(log_filename, "w") as f:
        f.write("True_Class,Predicted_Class,File_Path\n")
        
        with torch.no_grad():
            idx_counter = 0  # Tracks our absolute position in the val_idx list
            
            for range_seq, vel_seq, az_seq, el_seq, batch_labels in val_loader:
                range_seq = range_seq.to(device)
                vel_seq = vel_seq.to(device)
                az_seq = az_seq.to(device)
                el_seq = el_seq.to(device)
                
                # Unified Forward Pass
                outputs = model(range_seq, vel_seq, az_seq, el_seq)
                _, predicted = torch.max(outputs.data, 1)
                
                # Compare each sample in the batch
                for i in range(len(batch_labels)):
                    true_label = batch_labels[i].item()
                    pred_label = predicted[i].item()
                    
                    if true_label != pred_label:
                        # Grab the original dataset index from our split
                        original_idx = val_idx[idx_counter]
                        
                        # Fetch the actual file path from the dataset
                        file_path, _ = full_dataset.samples[original_idx]
                        
                        # Get the human-readable class names
                        true_name = full_dataset.classes[true_label]
                        pred_name = full_dataset.classes[pred_label]
                        
                        # Write to log
                        f.write(f"{true_name},{pred_name},{file_path}\n")
                        
                    idx_counter += 1

    print(f"Saved misclassification log to '{log_filename}'.")

if __name__ == "__main__":
    main()