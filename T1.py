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
        df = pd.read_csv(csv_file)

        self.data = []
        self.labels = []

        for _, row in df.iterrows():
            target_label = row['CE']  # Store CE label

            lengths = row.drop(['id', 'area', 'CE']).dropna().values.astype(float)
            
            # Generate 2N dihedral permutations
            for i in range(num_fences):
                
                rotated = np.roll(lengths, i) #shift all points by i positions
                
                rotated_reversed = rotated[::-1] # reverse array
                
                rotated_l0 = rotated[i]
                rotated_reversed_l0 = rotated_reversed[i]

                rotated = np.delete(rotated, i)
                rotated_reversed = np.delete(rotated_reversed, i)
                
                # Normalize
                norm_rotated = np.log1p(rotated / rotated_l0) 
                norm_rotated_reversed = np.log1p(rotated_reversed / rotated_reversed_l0)
                
                # Add data to the arrays
                self.data.append(norm_rotated)
                self.labels.append(target_label)

                self.data.append(norm_rotated_reversed)
                self.labels.append(target_label)

        # Convert to tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

  # Main Neural Network:

class FenceClassifier(nn.Module):
    def __init__(self, num_fences):
        super(FenceClassifier, self).__init__()
        self.fc1 = nn.Linear(num_fences - 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = F.leaky_relu(self.fc3(x), 0.1)
        x = F.leaky_relu(self.fc4(x), 0.1)
        x = F.leaky_relu(self.fc5(x), 0.1)
        x = F.leaky_relu(self.fc6(x), 0.1)
        return torch.sigmoid(self.out(x))

  # Train the NN:

def train_model(model, train_loader, val_loader, epochs=161):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels) # Compute loss
            loss.backward() # Backpropogation
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = correct / total * 100
        val_accuracies.append(val_accuracy)
        
        scheduler.step(avg_val_loss) # Scheduler (dynamic learning rate)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    return train_losses, val_losses, val_accuracies

  # Predict for Kaggle:

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
            row_lengths = row.iloc[1:10].values  # Extract all 9 fences

        # Normalize input
        row_lengths = row_lengths.astype(float)

        row_lengths_0 = row_lengths[0]

        row_lengths = np.delete(row_lengths, 0)

        transformed = np.log1p(row_lengths / row_lengths_0) # Dimensionless and log transformed

        # Convert to tensor
        input_tensor = torch.tensor(transformed, dtype=torch.float32).unsqueeze(0)

        # Select model and predict
        model = models[num_fences]
        model.eval()
        with torch.no_grad():
            predictions_probs = model(input_tensor).squeeze().numpy()
            prediction = (predictions_probs > 0.5).astype(int)  # Convert to 0 or 1
        
        # Store result
        predictions.append({'id': test_ids[i], 'prediction': prediction})
    
    # Save predictions
    output_df = pd.DataFrame(predictions)
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

  # Plot Graphs:

def plot_loss_curve(train_losses, val_losses, val_accuracies, num_fences):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves for {num_fences} Fences')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy for {num_fences} Fences')
    plt.legend()

    plt.tight_layout()
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
        model = FenceClassifier(num_fences)
        train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader)

        # Store trained models
        models[num_fences] = model  
        
        # Graphs
        plot_loss_curve(train_losses, val_losses, val_accuracies, num_fences)

        print(f"Avg Train Loss for {num_fences} fences: {np.mean(train_losses):.4f}")
        print(f"Avg Val Loss for {num_fences} fences: {np.mean(val_losses):.4f}")
        print(f"Final Val Accuracy for {num_fences} fences: {val_accuracies[-1]:.2f}%")

    # Kaggle submission
    predict(models, 'kaggle_hidden_test_fences.csv', 'submission_combined.csv')

if __name__ == "__main__":
    main()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
