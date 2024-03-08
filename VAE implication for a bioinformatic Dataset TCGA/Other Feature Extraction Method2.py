import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# Split data into features (X) and target (y)
X = Mrna2_df.drop("OS_STATUS", axis=1)
y = Mrna2_df["OS_STATUS"]

# Standardize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split the scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")




