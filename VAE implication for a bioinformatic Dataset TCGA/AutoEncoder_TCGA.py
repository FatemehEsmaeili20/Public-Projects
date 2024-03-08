import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # 2 * latent_dim for mean and log variance
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu, logvar = x_encoded.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_decoded = self.decoder(z)
        return x_decoded, mu, logvar

# Load and preprocess the dataset (data_clinical_patient.txt)
data = {}
f = open("data_clinical_patient.txt", "r")
for idx, line in enumerate(f):
        if idx == 4:
            header_cols_patients = line.strip().split('\t')
            data = {col: [] for col in header_cols_patients}
        elif idx > 4:
            x = line.split('\t')
            for i in range(len(header_cols_patients)):
                data[header_cols_patients[i]].append(x[i].strip())
patients_df = pd.DataFrame.from_dict(data)

# Load and preprocess the dataset (data_mrna_seq_rpkm.txt)
data = {}
f = open("data_mrna_seq_rpkm.txt", "r")
header = f.readline()
header_cols_mrna = header.split()
for col in header_cols_mrna:
    data[col] = []

for l in f:
    x = l.split('\t')
    for i in range(len(header_cols_mrna)):
        data[header_cols_mrna[i]].append(x[i].strip())
Mrna_df = pd.DataFrame.from_dict(data)

os_column_name = 'OS_STATUS'
patients_df[os_column_name] = patients_df[os_column_name].apply(lambda x: 1 if '1:DECEASED' in x else 0)
patients_df.head()

# Load clinical data
clinical_data = patients_df[["PATIENT_ID", "OS_STATUS"]]
clinical_data.loc[:, "OS_STATUS"] = clinical_data.loc[:, "OS_STATUS"].astype(int)

Mrna_df.drop(["Entrez_Gene_Id"], axis=1, inplace=True)
Mrna_df = Mrna_df.T
# Create a new DataFrame using the first row as column names
column_names = Mrna_df.iloc[0, 0:].tolist()
index_names= Mrna_df.index
Mrna2_df = pd.DataFrame(data=Mrna_df.values[0:], columns=column_names, index= index_names)
# Remove row at index 0
Mrna2_df = Mrna2_df.drop(Mrna2_df.index[0])
# Create OS_STATUS column in mRNA data
Mrna2_df["OS_STATUS"] = 0

# Populate OS_STATUS column based on clinical data
for patient_id, os_status in zip(clinical_data["PATIENT_ID"], clinical_data["OS_STATUS"]):
    new_patient_id = f"{patient_id}-03"
    if new_patient_id in Mrna2_df.index:
        Mrna2_df.at[new_patient_id, "OS_STATUS"] = os_status

# Standardize features using StandardScaler
scaler = StandardScaler()
Mrna_scaled = scaler.fit_transform(Mrna2_df)

df = Mrna_scaled.astype('float32')
data_tensor = torch.tensor(df)
data_tensor.shape

print("Processed Data Tensor:")
print(data_tensor)
print(data_tensor.shape[0],data_tensor.shape[1])

# Initialize model and optimizer
input_dim = data_tensor.shape[1] # Specify the size of your input data
latent_dim = 50  # Specify the desired size of the latent space
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Enable gradient clipping
max_grad_norm = 1.0  # Set the maximum gradient norm threshold
nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# Convert your data into a PyTorch DataLoader
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# Define the loss function
criterion = nn.MSELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = batch[0]
        recon_batch, mu, logvar = model(inputs)
        mse_loss = criterion(recon_batch, inputs)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = mse_loss + kl_loss
        total_loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}")

