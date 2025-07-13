import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import os
import onnx
import joblib

# ✅ Corrected path to match your actual CSV file
DATA_PATH = r"C:\Users\shriv\OneDrive\Desktop\intel project\healthcare_kiosk\backend\database\symptoms_disease.csv"
MODEL_DIR = r"C:\Users\shriv\OneDrive\Desktop\intel project\healthcare_kiosk\backend\models"
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "symptom_model.onnx")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Vectorize the 'symptoms' text column
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["symptoms"]).toarray().astype(np.float32)

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(df["disease"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom Dataset
class SymptomDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SymptomDataset(X_train, y_train)
test_ds = SymptomDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

# Model definition
class SymptomClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SymptomClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = X.shape[1]
num_classes = len(np.unique(y))

model = SymptomClassifier(input_size, num_classes).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20

# Train
model.train()
for epoch in range(epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(labels.numpy())

print("\nClassification Report:\n")
print(classification_report(all_true, all_preds, target_names=le.classes_))

# Save ONNX
dummy_input = torch.randn(1, input_size, device=device)
torch.onnx.export(
    model, dummy_input, ONNX_MODEL_PATH,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    opset_version=11
)

print(f"\n✅ ONNX model exported to: {ONNX_MODEL_PATH}")

# Save Label Encoder and Vectorizer
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.pkl")

joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"✅ Vectorizer saved to: {VECTORIZER_PATH}")

joblib.dump(le.classes_, CLASSES_PATH)
print(f"✅ Classes saved to: {CLASSES_PATH}")
