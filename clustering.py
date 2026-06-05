from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def apply_kmeans(rfm):
    """
    Applies KMeans clustering on RFM data
    """
    # Select features
    rfm_values = rfm[['Recency', 'Frequency', 'Monetary']]

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(rfm_values)

    # Apply KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(scaled_data) + 1

    return rfm, kmeans, scaler
