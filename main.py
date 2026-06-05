import gradio as gr
import pandas as pd

from src.data_preprocessing import load_and_clean_data
from src.rfm_analysis import create_rfm, score_rfm
from src.clustering import apply_kmeans

print("Starting app...")

# -----------------------------
# Load & train once
# -----------------------------
df = load_and_clean_data("data/data.csv")
rfm = create_rfm(df)
rfm = score_rfm(rfm)
rfm, kmeans_model, scaler = apply_kmeans(rfm)

print("Model ready")

# -----------------------------
# Prediction function
# -----------------------------
def predict_customer(recency, frequency, monetary):
    input_df = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=["Recency", "Frequency", "Monetary"]
    )

    scaled = scaler.transform(input_df)
    cluster = kmeans_model.predict(scaled)[0] + 1

    return f"Customer belongs to Cluster {cluster}"

# -----------------------------
# Gradio UI
# -----------------------------
app = gr.Interface(
    fn=predict_customer,
    inputs=[
        gr.Number(label="Recency (days)"),
        gr.Number(label="Frequency"),
        gr.Number(label="Monetary value")
    ],
    outputs=["text"],
    title="🛒 E-commerce Customer Segmentation & Prediction",
    description="Enter RFM values to predict customer segment using KMeans clustering."
)

app.launch()
