import os
import random
import re
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import RadarGestureDataset
from model.one_d_tcn import GestureRecognitionNetwork 
from model.srdst import SRDST_Adapted_Network
from model.lstm import LSTM_Gesture_Network

def set_seed(seed=42):
    """Locks all random seeds for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # 1. Lock seeds immediately
    set_seed(42)
    
    print("="*50)
    print("mmWave Hand Gesture Recognition Pipeline (4-Feature Edition)")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Dataset FIRST to dynamically get num_classes ---
    print("\nLoading dataset...")
    full_dataset = RadarGestureDataset(data_dir="Data", seq_length=40)
    labels = [sample[1] for sample in full_dataset.samples]
    
    NUM_CLASSES = len(full_dataset.classes)
    print(f"Detected {NUM_CLASSES} classes: {full_dataset.classes}")

    # --- 3. Model Selection UI ---
    print("\nAvailable Models:")
    print("1: FMCW Lightweight (DS-TCN + ECA)         [UPDATED FOR 4 INPUTS]")
    print("2: SRDST Adapted (Dual-Stream Transformer) [UPDATED FOR 4 INPUTS]")
    print("3: LSTM (Grobelny & Narbudowicz)           [UPDATED FOR 4 INPUTS]")
    
    choice = input("Enter the number of the model to train (1, 2, or 3): ").strip()

    if choice == '3':
        print(f"Initializing LSTM Model for {NUM_CLASSES} classes...")
        model = LSTM_Gesture_Network(num_classes=NUM_CLASSES).to(device)
        save_filename = "best_lstm_model_v51_rs_01.pth"
        model_name = "LSTM"
    elif choice == '2':
        print(f"Initializing SRDST Adapted Model for {NUM_CLASSES} classes...")
        model = SRDST_Adapted_Network(num_classes=NUM_CLASSES).to(device)
        save_filename = "best_srdst_model_v51_rs_01.pth"
        model_name = "SRDST"
    else:
        if choice != '1':
            print("Invalid input. Defaulting to FMCW Lightweight Model...")
        print(f"Initializing FMCW Lightweight Model for {NUM_CLASSES} classes...")
        model = GestureRecognitionNetwork(num_classes=NUM_CLASSES).to(device)
        save_filename = "best_fmcw_model_v51_rs_01.pth"
        model_name = "TCN"

    # --- 4. Hyperparameters ---
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 16

    # --- 5. Split the Dataset (Robust File-Checking Method) ---
    def get_file_number(filepath):
        """Extracts the first number found in the filename."""
        filename = os.path.basename(str(filepath))
        nums = re.findall(r'\d+', filename)
        return int(nums[0]) if nums else None
    
    # 1. Cari semua ID yang BENAR-BENAR ada di folder dataset Anda
    existing_ids = set()
    for filepath, _ in full_dataset.samples:
        file_num = get_file_number(filepath)
        if file_num is not None:
            existing_ids.add(file_num)

    # Fungsi pembantu untuk mengumpulkan ID yang valid dalam rentang tertentu
    def get_valid_pool(ranges, existing):
        pool = []
        for start, end in ranges:
            pool.extend([n for n in existing if start <= n <= end])
        return pool

    # 2. Kumpulkan ID berdasarkan Subjek (Sesuai Indeks di Tabel)
    pool_A = get_valid_pool([(1, 200), (701, 800)], existing_ids)
    pool_C = get_valid_pool([(601, 700)], existing_ids)
    pool_E = get_valid_pool([(451, 500)], existing_ids)
    pool_F = get_valid_pool([(251, 300)], existing_ids)
    pool_I = get_valid_pool([(321, 370)], existing_ids)
    pool_H = get_valid_pool([(426, 450)], existing_ids)
    
    pool_G = get_valid_pool([(201, 250), (801, 810)], existing_ids) # Khusus Val

    # 3. Lakukan Sampling sesuai "Jumlah Data" di Tabel
    # Menggunakan min() agar terhindar dari error jika data asli kurang dari target
    train_A = random.sample(pool_A, min(25, len(pool_A)))
    train_C = random.sample(pool_C, min(50, len(pool_C)))
    train_E = random.sample(pool_E, min(50, len(pool_E)))
    train_F = random.sample(pool_F, min(50, len(pool_F)))
    train_I = random.sample(pool_I, min(40, len(pool_I)))
    train_H = random.sample(pool_H, min(25, len(pool_H)))

    # Subjek G seluruhnya dialokasikan untuk Validation (Target: 60)
    val_G = random.sample(pool_G, min(60, len(pool_G)))

    # 4. Gabungkan ke dalam TRAIN_POOL dan VAL_POOL master
    TRAIN_POOL = set(train_A + train_C + train_E + train_F + train_I + train_H)
    VAL_POOL = set(val_G)

    # 5. Distribusikan indeks dataset berdasarkan ID filenya
    train_idx = []
    val_idx = []

    for i, (filepath, _) in enumerate(full_dataset.samples):
        file_num = get_file_number(filepath)
        
        if file_num is None:
            continue
            
        if file_num in VAL_POOL:
            val_idx.append(i)
        elif file_num in TRAIN_POOL:
            train_idx.append(i)

    # 6. Buat PyTorch subsets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    print(f"Data Split - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    if len(train_dataset) + len(val_dataset) > 0:
        print(f"Validation percentage: {(len(val_dataset) / (len(train_dataset) + len(val_dataset))) * 100:.2f}%\n")
    else:
        print("Error: Dataset appears to be empty based on your ID rules!\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 6. Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 7. Training Loop ---
    best_val_acc = 0.0

    history_train_loss, history_val_loss = [], []
    history_train_acc, history_val_acc = [], []

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

            optimizer.zero_grad()
            
            # ---> UNIFIED FORWARD PASS
            outputs = model(range_seq, vel_seq, az_seq, el_seq)
                
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc= f"{100.0 * correct_train / total_train:.2f}%")

        avg_train_loss = train_loss / len(train_loader)
        train_acc = (correct_train / total_train) * 100

        history_train_loss.append(avg_train_loss)
        history_train_acc.append(train_acc)

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

        history_val_loss.append(avg_val_loss)
        history_val_acc.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_filename)
            print(f"  -> Saved new best model to {save_filename}!")

    print(f"\nTraining Finished! Best Validation Accuracy for {model_name}: {best_val_acc:.2f}%")

    # ==========================================
    # DEBUGGING: Training Curves (Loss & Accuracy)
    # ==========================================
    print(f"\n--- Generating Learning Curves for {model_name} ---")
    epochs_range = range(1, num_epochs + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_train_loss, label='Train Loss', color='blue')
    plt.plot(epochs_range, history_val_loss, label='Val Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_train_acc, label='Train Accuracy', color='blue')
    plt.plot(epochs_range, history_val_acc, label='Val Accuracy', color='orange')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    curves_filename = f"debug_learning_curves_51_rs_01_{model_name}.png"
    plt.savefig(curves_filename, dpi=300)
    print(f"Saved learning curves to '{curves_filename}'. Please review it!")

    # ==========================================
    # DEBUGGING: Post-Training Confusion Matrix
    # ==========================================
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
    cm_filename = f"debug_confusion_matrix_51_rs_01_{model_name}.png"
    plt.savefig(cm_filename, dpi=300)
    print(f"Saved '{cm_filename}'. Please review it!")

    # ==========================================
    # DEBUGGING: Misclassified File Logger
    # ==========================================
    print("\n--- Diagnosing Misclassified Files ---")
    
    # Ensure the model is in eval mode and best weights are loaded
    model.load_state_dict(torch.load(save_filename))
    model.eval()

    log_filename = f"misclassified_log_51_rs_01_{model_name}.csv"
    
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