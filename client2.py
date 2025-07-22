import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("/home/gul/nsl_kdd")
from sklearn.linear_model import LogisticRegression
from nsl_kdd_preprocess import load_nsl_kdd
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ==== CONFIGURE ====
client_id = 1         # Change to 1 or 2 for other clients
client_prefix = "client2"   # Update to "client2" etc. for other clients
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# ==== LOAD DATA ====
X_train, X_test, y_train, y_test, client_original_labels, attack_label_mapping = load_nsl_kdd(
    "/home/gul/nsl_kdd/KDDTrain+_balanced.txt",
    "/home/gul/nsl_kdd/KDDTest+.txt",
    client_id=client_id,
    num_clients=3
)

# ==== ATTACK TYPE DISTRIBUTION (CSV & PNG) ====
unique, counts = np.unique(client_original_labels, return_counts=True)
df_dist = pd.DataFrame({
    "AttackType": [attack_label_mapping[idx] for idx in unique],
    "Count": counts
})
df_dist.to_csv(f"{result_dir}/{client_prefix}_attack_type_distribution.csv", index=False)

plt.figure(figsize=(14,6))
plt.bar([attack_label_mapping[idx] for idx in unique], counts)
plt.title(f"{client_prefix.capitalize()} - Attack Type Distribution (Original Labels)")
plt.xlabel("Attack Type")
plt.ylabel("Count")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.savefig(f"{result_dir}/{client_prefix}_attack_type_distribution.png")
plt.close()

# ==== TRAIN MODEL ====
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

# ==== CONFUSION MATRIX (PNG) ====
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
disp.plot()
plt.title(f"{client_prefix.capitalize()} - Confusion Matrix")
plt.savefig(f"{result_dir}/{client_prefix}_confusion_matrix.png")
plt.close()

# ==== CLASSIFICATION REPORT (TXT) ====
report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])
with open(f"{result_dir}/{client_prefix}_classification_report.txt", "w") as f:
    f.write(report)

print(f"\nClassification Report for {client_prefix.capitalize()}:\n{report}\n")

# ==== FEDERATED CLIENT ====
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print(f"{client_prefix.capitalize()} - get_parameters")
        return [clf.coef_, clf.intercept_]

    def fit(self, parameters, config):
        print(f"{client_prefix.capitalize()} - fit")
        clf.coef_, clf.intercept_ = parameters
        clf.fit(X_train, y_train)
        return [clf.coef_, clf.intercept_], len(X_train), {}

    def evaluate(self, parameters, config):
        print(f"{client_prefix.capitalize()} - evaluate")
        clf.coef_, clf.intercept_ = parameters
        loss = 1.0 - clf.score(X_test, y_test)
        accuracy = clf.score(X_test, y_test)
        return float(loss), len(X_test), {"accuracy": float(accuracy)}

# ==== START CLIENT ====
fl.client.start_numpy_client(server_address="13.60.24.113:8080", client=FlowerClient())

