import warnings
# This completely mutes all warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import optuna
from optuna.trial import TrialState
from tqdm import tqdm # Standard tqdm for local terminal use

# Import your custom modules natively
from dataset2 import RadarGestureDataset
from model.one_d_tcn_v2 import GestureRecognitionNetwork

print("Loading dataset into memory...")
# Point directly to your local data folder
DATA_DIR = "Data_51cm" 

full_dataset = RadarGestureDataset(data_dir=DATA_DIR, seq_length=40)
labels = [sample[1] for sample in full_dataset.samples]
NUM_CLASSES = len(full_dataset.classes)

train_idx, val_idx = train_test_split(
    range(len(full_dataset)), test_size=0.2, stratify=labels, random_state=42
)
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

# GPU-Optimized DataLoaders (adjust batch_size down if your local machine runs out of memory)
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

def objective(trial):
    # ==========================================
    # STREAMLINED HYPERPARAMETER SEARCH SPACE
    # ==========================================
    # 1. THE HEAVY HITTERS (Optuna tunes these)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    # 2. THE LOCKED DEFAULTS (Fixed values to save Optuna time)
    temporal_dropout = 0.15 
    fc_dropout = 0.5        

    model = GestureRecognitionNetwork(
        num_classes=NUM_CLASSES, 
        temporal_dropout=temporal_dropout, 
        fc_dropout=fc_dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    EPOCHS = 40 

    for epoch in range(EPOCHS):
        # ==========================================
        # TRAINING PHASE
        # ==========================================
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loop = tqdm(train_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

        for range_seq, vel_seq, az_seq, el_seq, batch_labels in train_loop:
            range_seq = range_seq.to(device)
            vel_seq = vel_seq.to(device)
            az_seq = az_seq.to(device)
            el_seq = el_seq.to(device)
            batch_labels = batch_labels.to(device)

            # --- Data Augmentation ---
            scale = torch.empty(batch_labels.size(0), 1, 1, device=device).uniform_(0.8, 1.2)
            range_seq *= scale
            vel_seq *= scale

            noise_level_rv = 0.02
            noise_level_azel = 0.005

            range_seq += torch.randn_like(range_seq) * noise_level_rv
            vel_seq += torch.randn_like(vel_seq) * noise_level_rv
            az_seq += torch.randn_like(az_seq) * noise_level_azel
            el_seq += torch.randn_like(el_seq) * noise_level_azel

            shift = torch.randint(-4, 5, (1,)).item()
            if shift != 0:
                range_seq = torch.roll(range_seq, shifts=shift, dims=2)
                vel_seq = torch.roll(vel_seq, shifts=shift, dims=2)
                az_seq = torch.roll(az_seq, shifts=shift, dims=2)
                el_seq = torch.roll(el_seq, shifts=shift, dims=2)

            optimizer.zero_grad()
            outputs = model(range_seq, vel_seq, az_seq, el_seq)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * range_seq.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()

            train_loop.set_postfix(
                loss=f"{loss.item():.4f}", 
                avg_loss=f"{running_loss/total_train:.4f}", 
                acc=f"{(correct_train/total_train)*100:.2f}%"
            )

        # ==========================================
        # VALIDATION PHASE
        # ==========================================
        model.eval()
        correct_val, total_val = 0, 0
        val_loop = tqdm(val_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)

        with torch.no_grad():
            for range_seq, vel_seq, az_seq, el_seq, batch_labels in val_loop:
                range_seq = range_seq.to(device)
                vel_seq = vel_seq.to(device)
                az_seq = az_seq.to(device)
                el_seq = el_seq.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(range_seq, vel_seq, az_seq, el_seq)
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_labels.size(0)
                correct_val += (predicted == batch_labels).sum().item()

                val_loop.set_postfix(acc=f"{(correct_val/total_val)*100:.2f}%")

        val_acc = correct_val / total_val

        # ==========================================
        # OPTUNA PRUNING LOGIC
        # ==========================================
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc

if __name__ == "__main__":
    print("Initializing Streamlined Local Optuna Study...")
    
    # Adjusted Pruner for a 15-trial search (forgives the first 3 trials)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10)
    
    # LOCAL DATABASE: Creates 'optuna_study.db' right next to your script
    DB_PATH = "sqlite:///optuna_study.db"
    STUDY_NAME = "tcn_tuning_study"
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_PATH,
        direction="maximize", 
        pruner=pruner,
        load_if_exists=True
    )
    
    print(f"Worker connected! Current completed trials in database: {len(study.trials)}")
    
    # Run the streamlined 15 trials
    study.optimize(objective, n_trials=15)

    print("\nSearch complete!")
    
    # Print the best result
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    if len(complete_trials) > 0:
        print("\nBest overall trial:")
        trial = study.best_trial
        print(f"  Validation Accuracy: {trial.value * 100:.2f}%")
        print(f"  Best Learning Rate: {trial.params['lr']}")
        print(f"  Best Weight Decay: {trial.params['weight_decay']}")
        print("  (Locked Parameters: temporal_dropout=0.15, fc_dropout=0.5)")