import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
    # Make sure "Data" is the folder containing your 7 new gesture classes
    full_dataset = RadarGestureDataset(data_dir="Data", seq_length=40)
    labels = [sample[1] for sample in full_dataset.samples]
    
    NUM_CLASSES = len(full_dataset.classes)
    print(f"Detected {NUM_CLASSES} classes: {full_dataset.classes}")

    # --- 2. Model Selection UI ---
    print("\nAvailable Models:")
    print("1: FMCW Lightweight (DS-TCN + ECA) [UPDATED FOR 4 INPUTS]")
    print("2: SRDST Adapted (Dual-Stream Transformer) [REQUIRES 3 INPUTS]")
    print("3: LSTM (Grobelny & Narbudowicz) [REQUIRES 3 INPUTS]")
    
    choice = input("Enter the number of the model to train (1, 2, or 3): ").strip()

    if choice == '3':
        print(f"Initializing LSTM Model for {NUM_CLASSES} classes...")
        model = LSTM_Gesture_Network(num_classes=NUM_CLASSES).to(device)
        save_filename = "best_lstm_model.pth"
    elif choice == '2':
        print(f"Initializing SRDST Adapted Model for {NUM_CLASSES} classes...")
        model = SRDST_Adapted_Network(num_classes=NUM_CLASSES).to(device)
        save_filename = "best_srdst_model.pth"
    else:
        if choice != '1':
            print("Invalid input. Defaulting to FMCW Lightweight Model...")
        print(f"Initializing FMCW Lightweight Model for {NUM_CLASSES} classes...")
        model = GestureRecognitionNetwork(num_classes=NUM_CLASSES).to(device)
        save_filename = "best_fmcw_model.pth"

    # --- 3. Hyperparameters ---
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 16

    # --- 4. Split the Dataset ---
    # 6:2:2 Stratified Split
    train_idx, temp_idx = train_test_split(range(len(full_dataset)), test_size=0.4, stratify=labels, random_state=42)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=42)
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    print(f"Data Split - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 5. Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- 6. Training Loop ---
    best_val_acc = 0.0
    print("\nStarting Training...")
    
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
            
            # SAFETY CATCH: Only the TCN (choice 1) was updated for 4 inputs
            if choice == '1' or choice not in ['2', '3']:
                outputs = model(range_seq, vel_seq, az_seq, el_seq)
            else:
                # If using older unpatched models, just pass the original 3 features
                outputs = model(range_seq, vel_seq, az_seq)
                
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
                
                if choice == '1' or choice not in ['2', '3']:
                    outputs = model(range_seq, vel_seq, az_seq, el_seq)
                else:
                    outputs = model(range_seq, vel_seq, az_seq)
                    
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

    print(f"\nTraining Finished! Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()