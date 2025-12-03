from model import TAModel

import pandas as pd
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    Dataset
)

import kagglehub
from kagglehub import KaggleDatasetAdapter

import warnings
warnings.filterwarnings('ignore')

# Let's load the dataset
filepath = "Titanic-Dataset.csv"
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "yasserh/titanic-dataset",
    filepath
)

# Now let's clean it up
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']]
df = df.replace({"male": 1, "female": 2})
df['Age'] = df['Age'].fillna(df['Age'].mean())


categorical_features = ['Pclass', 'Sex']
continuous_features = ['Age', 'SibSp', 'Parch']

df_cat = pd.get_dummies(df[categorical_features], columns=categorical_features)
df_cont = df[continuous_features].fillna(df[continuous_features].mean())

features = pd.concat([df_cat, df_cont], axis=1)
features = features.astype(float)
features = torch.tensor(features.values, dtype=torch.float32)
targets = torch.tensor(df['Survived'].values, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(features, targets)
dataloader = DataLoader(dataset, batch_size=len(features), shuffle=True)
input_dim = features.shape[1]

cat_indices = list(range(df_cat.shape[1]))
cont_indices = list(range(df_cat.shape[1], input_dim))


def loss_fn(x_hat, x):
    loss_cat = nn.MSELoss()(x_hat[:, cat_indices], x[:, cat_indices])
    loss_cont = nn.MSELoss()(x_hat[:, cont_indices], x[:, cont_indices])
    return loss_cat + loss_cont


# Now lets setup the trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TAModel(input_dim=8, latent_dim=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 1000

bce = nn.BCELoss()

for epoch in range(num_epochs):
    total_loss = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        x_hat, y_hat = model(x)
        ae_loss = loss_fn(x_hat, x)
        cls_loss = bce(y_hat, y)

        loss = ae_loss + cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {avg_loss:.4f}")
torch.save(model, "saved_model.mdl")
