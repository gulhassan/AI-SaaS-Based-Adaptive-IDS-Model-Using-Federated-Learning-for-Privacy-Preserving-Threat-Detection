import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
sys.path.append("/home/gul/USNW")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from unsw_nb15_preprocess import load_unsw_nb15

client_id = 0
client_prefix = "client1"
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

X_train, X_test, y_train, y_test, client_original_labels, attack_label_mapping = load_unsw_nb15(
    "/home/gul/USNW/UNSW_NB15_training-set.csv", "/home/gul/USNW/UNSW_NB15_testing-set.csv", client_id=client_id, num_clients=3)

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

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
disp.plot()
plt.title(f"{client_prefix.capitalize()} - Confusion Matrix")
plt.savefig(f"{result_dir}/{client_prefix}_confusion_matrix.png")
plt.close()

report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])
with open(f"{result_dir}/{client_prefix}_classification_report.txt", "w") as f:
    f.write(report)
print(f"\nClassification Report for {client_prefix.capitalize()}:\n{report}\n")

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

fl.client.start_numpy_client(server_address="13.60.24.113:8080", client=FlowerClient())

