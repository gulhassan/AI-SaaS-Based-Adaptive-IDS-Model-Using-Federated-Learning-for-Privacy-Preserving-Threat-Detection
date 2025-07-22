import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# All results are in this directory
result_dir = "results"

st.set_page_config(page_title="SaaS Federated IDS Demo", layout="wide")

st.title("SaaS-Based Federated IDS Dashboard")
st.write("**Demo: No real logs transferred. All client data remains private.**")

# List of clients (add or change as needed)
clients = ["Client 1", "Client 2", "Client 3"]
client_files = ["client1", "client2", "client3"]

st.sidebar.title("Select Organization")
selected = st.sidebar.selectbox("Choose a client to view:", clients)
sel_idx = clients.index(selected)
prefix = client_files[sel_idx]

# ---- Attack Type Distribution ----
st.header(f"{selected} - Attack Type Distribution")
csv_file = f"{result_dir}/{prefix}_attack_type_distribution.csv"
png_file = f"{result_dir}/{prefix}_attack_type_distribution.png"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["AttackType"], df["Count"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Count")
    st.pyplot(fig)
elif os.path.exists(png_file):
    st.image(png_file, caption="Attack Type Distribution")
else:
    st.info("No attack distribution data found for this client.")

# ---- Confusion Matrix ----
img_file = f"{result_dir}/{prefix}_confusion_matrix.png"
if os.path.exists(img_file):
    st.subheader("Confusion Matrix")
    st.image(img_file, width=400)
else:
    st.info("No confusion matrix found for this client.")

# ---- Classification Report ----
report_file = f"{result_dir}/{prefix}_classification_report.txt"
if os.path.exists(report_file):
    st.subheader("Classification Report")
    with open(report_file) as f:
        report = f.read()
    st.code(report)
else:
    st.info("No classification report found for this client.")

# ---- Global Model Accuracy (Optional) ----
accuracy_log_file = f"{result_dir}/accuracy_log.txt"
if os.path.exists(accuracy_log_file):
    with open(accuracy_log_file) as f:
        lines = f.readlines()
    st.subheader("Global Model Accuracy (by round)")
    st.code("".join(lines))
else:
    st.info("No accuracy log found.")

# ---- Simulate Alerts ----
st.sidebar.markdown("---")
if st.sidebar.button("Simulate Alert"):
    st.warning(f"{selected}: Intrusion Detected! (Simulated alert)")

st.sidebar.markdown("---")
st.sidebar.success("All client data is kept local. Only model updates are sent to the cloud.")

st.markdown("""
---
#### How does it work?
- Each client trains its model locally, shares only model updates with the SaaS server.
- The global model is improved collaboratively, benefiting all organizations.
- No raw traffic logs or confidential data are ever uploaded to the cloud.
""")

