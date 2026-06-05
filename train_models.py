# STEP 1: Import functions from src folder
from src.data_preprocessing import load_and_clean_data
from src.rfm_analysis import create_rfm, score_rfm
from src.clustering import apply_kmeans

# STEP 2: Load and clean raw data
print("Loading data...")
df = load_and_clean_data("data/data.csv")

# STEP 3: Create RFM table
print("Creating RFM table...")
rfm = create_rfm(df)
rfm = score_rfm(rfm)

# STEP 4: Apply KMeans clustering
print("Applying KMeans clustering...")
rfm, kmeans_model, scaler = apply_kmeans(rfm)

# STEP 5: Save outputs (CSV)
print("Saving RFM data...")
rfm.to_csv("models/rfm_with_clusters.csv")

print("✅ Training completed successfully")
print("📁 Output file created: models/rfm_with_clusters.csv")
