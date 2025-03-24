# Imports

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt 
import itertools
from itertools import permutations

# Obtain data, convert to tensors:

class FenceDataset(Dataset):
    def __init__(self, csv_file, num_fences):
        df = pd.read_csv(csv_file) # Get the csv file

        self.data = []
        self.labels = []

        for _, row in df.iterrows():
            target_label = row['area']  # Raw area
            
            lengths = row.drop(['id', 'area', 'CE']).dropna().values.astype(float) # Get the lengths as an array
            squared_lengths = lengths ** 2  # Square all lengths to match area units ([L]^2)
            
            # Generate 2N dihedral permutations
            for i in range(num_fences):
                rotated = np.roll(squared_lengths, i)  # Shift all points by i positions
                rotated_reversed = rotated[::-1]  # Reverse array
                
                # Add data to the arrays
                self.data.append(rotated)
                self.labels.append(target_label)

                self.data.append(rotated_reversed)
                self.labels.append(target_label)

        # Convert to tensors
        self.data = np.array(self.data, dtype=np.float32)
        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.float32) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Main Neural Network:

class FenceRegressor(nn.Module):
    def __init__(self, num_fences):
        super(FenceRegressor, self).__init__()
        self.fc1 = nn.Linear(num_fences, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return self.out(x)

# MAPE Loss:

def mape_loss(output, target):
    return torch.mean(torch.abs((target - output) / target)) * 100

# Train the NN:

def train_model(model, train_loader, val_loader, epochs=121):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze() 
            loss = mape_loss(outputs, labels)  # Compute loss
            loss.backward() # Backpropogation
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation 
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze()
                loss = mape_loss(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss) # Scheduler (dynamic learning rate)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Train MAPE: {avg_train_loss:.4f}%, Val MAPE: {avg_val_loss:.4f}%")

    return train_losses, val_losses

# Predict for kaggle:

def predict(models, test_file, output_file):
    df_test = pd.read_csv(test_file)

    predictions = []
    test_ids = df_test['id'].values  # Get test IDs

    for i in range(len(df_test)):
        row = df_test.iloc[i]
        
        # Determine fence case
        if row[[6, 7, 8, 9]].isna().all():  # Case: 5 fences
            num_fences = 5
            row_lengths = row.iloc[1:6].values  # Extract first 5 fences
        elif row[[8, 9]].isna().all() and row[[6, 7]].notna().all():  # Case: 7 fences
            num_fences = 7
            row_lengths = row.iloc[1:8].values  # Extract first 7 fences
        else:  # Case: 9 fences
            num_fences = 9
            row_lengths = row.iloc[1:10].values # Extract all 9 fences

        # Square and normalize input
        row_lengths = row_lengths.astype(float)
        squared_lengths = row_lengths ** 2

        # Convert to tensor
        input_tensor = torch.tensor(squared_lengths, dtype=torch.float32).unsqueeze(0)

        # Select model and predict
        model = models[num_fences]
        model.eval()
        with torch.no_grad():
            pred_area = model(input_tensor).squeeze().item()
            # Store result
            predictions.append({'id': test_ids[i], 'prediction': pred_area})
    
    # Save predictions
    output_df = pd.DataFrame(predictions)
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Plot Graphs:

def plot_loss_curve(train_losses, val_losses, num_fences):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training MAPE', color='blue')
    plt.plot(val_losses, label='Validation MAPE', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('MAPE (%)')
    plt.title(f'MAPE Curves for {num_fences} Fences')
    plt.legend()
    plt.show()

# Main method:

def main():
    models = {}

    for num_fences in [5, 7, 9]:
        # Load dataset
        dataset = FenceDataset(f'kaggle_train_{num_fences}_fences.csv', num_fences)

        # Split into 80/20 training, testing
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Initialize and train model
        model = FenceRegressor(num_fences)
        train_losses, val_losses = train_model(model, train_loader, val_loader)

        # Store trained model
        models[num_fences] = model  

        # Graphs
        plot_loss_curve(train_losses, val_losses, num_fences)

        print(f"Avg Train MAPE for {num_fences} fences: {np.mean(train_losses):.4f}%")
        print(f"Avg Val MAPE for {num_fences} fences: {np.mean(val_losses):.4f}%")

    # Kaggle submission
    predict(models, 'kaggle_hidden_test_fences.csv', 'submission_T2.csv')

if __name__ == "__main__":
    main()
