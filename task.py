# -----------------------------------------
# 📦 1. Import Required Libraries
# -----------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Optional: Upload File (use this for Jupyter or Google Colab)
try:
    from google.colab import files
    uploaded = files.upload()
except:
    print("Not running in Google Colab. Make sure the CSV is in the working directory.")

# -----------------------------------------
# 📄 2. Load the Dataset
# -----------------------------------------
try:
    df = pd.read_csv("customer_data.csv")
    print("✅ File loaded successfully!")
except FileNotFoundError:
    print("❌ File not found. Please check the filename or path.")
    raise

# -----------------------------------------
# 🔍 3. Inspect the Data
# -----------------------------------------
print("\n🔹 Dataset Shape:", df.shape)
print("🔹 Missing values:\n", df.isnull().sum())
print("🔹 Data types:\n", df.dtypes)
print("🔹 Summary:\n", df.describe())

# Drop 'Customer ID' (not needed for clustering)
df_clean = df.drop("Customer ID", axis=1)

# -----------------------------------------
# 🧼 4. Preprocessing (Scaling)
# -----------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)

# -----------------------------------------
# 📊 5. Elbow Method to Find Optimal Clusters
# -----------------------------------------
wcss = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, wcss, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# -----------------------------------------
# 🧪 6. Silhouette Score (Optional, for better judgment)
# -----------------------------------------
print("\n📈 Silhouette Scores:")
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    print(f"k = {k} → Silhouette Score = {score:.4f}")

# -----------------------------------------
# 🧠 7. Apply K-Means (Set k based on Elbow)
# -----------------------------------------
optimal_k = 5  # Change this based on Elbow/Silhouette result
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_data)

# -----------------------------------------
# 🖼️ 8. PCA for 2D Visualization
# -----------------------------------------
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
df["PCA1"] = pca_components[:, 0]
df["PCA2"] = pca_components[:, 1]

# Plotting the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100)
plt.title("Customer Clusters (PCA 2D View)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

# -----------------------------------------
# 🔍 9. Pair Plot for Feature Insights
# -----------------------------------------
sns.pairplot(df, vars=["Age", "Annual Income", "Spending Score"], hue="Cluster", palette="Set2")
plt.suptitle("Feature Relationships by Cluster", y=1.02)
plt.show()

# -----------------------------------------
# 💾 10. Save the Clustered Dataset
# -----------------------------------------
df.to_csv("clustered_customers.csv", index=False)
print("\n✅ Clustered dataset saved as 'clustered_customers.csv'")
