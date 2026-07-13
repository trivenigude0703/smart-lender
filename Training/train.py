import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'loan_prediction.csv')
print(f"Reading dataset from {data_path}...")
data = pd.read_csv(data_path)
print("Shape:", data.shape)

# ─────────────────────────────────────────────────
# 2. Handling Categorical Values (exact mapping from course)
# ─────────────────────────────────────────────────
data['Gender']        = data['Gender'].map({'Female': 1, 'Male': 0})
data['Property_Area'] = data['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
data['Married']       = data['Married'].map({'Yes': 1, 'No': 0})
data['Education']     = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Loan_Status']   = data['Loan_Status'].map({'Y': 1, 'N': 0})

# ─────────────────────────────────────────────────
# 3. Handling Missing Values
# ─────────────────────────────────────────────────
print("\nMissing values before:\n", data.isnull().sum())

data['Gender']          = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married']         = data['Married'].fillna(data['Married'].mode()[0])

# Replace '+' in Dependents then fill NA
data['Dependents'] = data['Dependents'].str.replace('+', '', regex=False)
data['Dependents']      = data['Dependents'].fillna(data['Dependents'].mode()[0])

data['Self_Employed']   = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['LoanAmount']      = data['LoanAmount'].fillna(data['LoanAmount'].mode()[0])
data['Loan_Amount_Term']= data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data['Credit_History']  = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

print("\nMissing values after:\n", data.isnull().sum())
data.info()

# ─────────────────────────────────────────────────
# 4. Cast float columns to int64
# ─────────────────────────────────────────────────
data['Gender']             = data['Gender'].astype('int64')
data['Married']            = data['Married'].astype('int64')
data['Dependents']         = data['Dependents'].astype('int64')
data['Self_Employed']      = data['Self_Employed'].astype('int64')
data['CoapplicantIncome']  = data['CoapplicantIncome'].astype('int64')
data['LoanAmount']         = data['LoanAmount'].astype('int64')
data['Loan_Amount_Term']   = data['Loan_Amount_Term'].astype('int64')
data['Credit_History']     = data['Credit_History'].astype('int64')

data.info()

# ─────────────────────────────────────────────────
# 5. EDA Plots
# ─────────────────────────────────────────────────
plt.style.use('fivethirtyeight')
os.makedirs('plots', exist_ok=True)

plt.figure(figsize=(12, 5))
plt.subplot(121)
sns.histplot(data['ApplicantIncome'], color='r', kde=True)
plt.title('Applicant Income Distribution')
plt.subplot(122)
sns.histplot(data['Credit_History'], kde=True)
plt.title('Credit History Distribution')
plt.tight_layout()
plt.savefig('plots/univariate_continuous.png')
plt.close()

plt.figure(figsize=(18, 4))
plt.subplot(1, 4, 1)
sns.countplot(x=data['Gender'])
plt.title('Gender Count')
plt.subplot(1, 4, 2)
sns.countplot(x=data['Education'])
plt.title('Education Count')
plt.subplot(1, 4, 3)
sns.countplot(x=data['Married'])
plt.title('Married Count')
plt.subplot(1, 4, 4)
sns.countplot(x=data['Self_Employed'])
plt.title('Self Employed Count')
plt.tight_layout()
plt.savefig('plots/univariate_categorical.png')
plt.close()

# ─────────────────────────────────────────────────
# 6. Split Features and Target
# ─────────────────────────────────────────────────
x = data.drop(columns=['Loan_ID', 'Loan_Status'])
y = data['Loan_Status']

print("\nClass distribution before SMOTE:\n", y.value_counts())

# ─────────────────────────────────────────────────
# 7. Apply SMOTE on full dataset (before splitting)
# ─────────────────────────────────────────────────
smote = SMOTE(random_state=42)
x_bal, y_bal = smote.fit_resample(x, y)
print("\nClass distribution after SMOTE:\n", y_bal.value_counts())

names = x_bal.columns

# ─────────────────────────────────────────────────
# 8. Scaling on balanced dataset
# ─────────────────────────────────────────────────
sc = StandardScaler()
x_bal_scaled = sc.fit_transform(x_bal)
x_bal = pd.DataFrame(x_bal_scaled, columns=names)

# ─────────────────────────────────────────────────
# 9. Train/Test Split (test_size=0.33)
# ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    x_bal, y_bal, test_size=0.33, random_state=42
)
print("\nX_train shape:", X_train.shape)
print("X_test  shape:", X_test.shape)

# ─────────────────────────────────────────────────
# 10. Model Training & Evaluation
# ─────────────────────────────────────────────────
models = {
    'Decision Tree':      DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting':  GradientBoostingClassifier(random_state=42),
    'Random Forest':      RandomForestClassifier(random_state=42),
}

best_model_obj  = None
best_f1         = 0
best_model_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    print(f"  Accuracy: {acc:.4f}  |  F1-Score: {f1:.4f}")
    print(classification_report(y_test, preds))
    if f1 > best_f1:
        best_f1         = f1
        best_model_obj  = model
        best_model_name = name

print(f"\nBest Model → {best_model_name}  F1={best_f1:.4f}")

# ─────────────────────────────────────────────────
# 11. Pickle model and scaler into Flask/
# ─────────────────────────────────────────────────
flask_dir = os.path.join(os.path.dirname(__file__), '..', 'Flask')
os.makedirs(flask_dir, exist_ok=True)

rf_model = models['Random Forest']

print("\nSaving rdf.pkl ...")
with open(os.path.join(flask_dir, 'rdf.pkl'), 'wb') as f:
    pickle.dump(rf_model, f)

print("Saving scale1.pkl ...")
with open(os.path.join(flask_dir, 'scale1.pkl'), 'wb') as f:
    pickle.dump(sc, f)

with open(os.path.join(flask_dir, 'scale1(1).pkl'), 'wb') as f:
    pickle.dump(sc, f)

print("\nAll done! Pickles saved to Flask/")
