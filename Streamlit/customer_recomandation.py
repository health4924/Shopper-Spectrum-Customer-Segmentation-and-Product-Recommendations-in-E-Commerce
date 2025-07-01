import streamlit as st
import numpy as np
import pickle

# ------------------------------
# Load Pickled Models and Data
# ------------------------------

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('product_similarity.pkl', 'rb') as f:
    product_similarity_df = pickle.load(f)

# ------------------------------
# Product Recommendation Function
# ------------------------------

def get_similar_products(product_name, top_n=5):
    if product_name not in product_similarity_df.columns:
        return []
    similar_scores = product_similarity_df[product_name].sort_values(ascending=False)
    return similar_scores.iloc[1:top_n+1].index.tolist()

# ------------------------------
# Streamlit App
# ------------------------------

st.set_page_config(page_title="Shopper Spectrum", layout="wide")
st.title("Shopper Spectrum: Customer Segmentation & Product Recommendations")

tab1, tab2 = st.tabs(["Product Recommender", "Customer Segmentation"])

# ---------------------------------
# Product Recommendation Tab
# ---------------------------------
with tab1:
    st.header("Product Recommendation Engine")

    product_list = sorted(product_similarity_df.columns.tolist())
    selected_product = st.selectbox("Select a Product for Recommendation", product_list)

    if st.button("Get Recommendations"):
        if selected_product:
            recommendations = get_similar_products(selected_product)
            if recommendations:
                st.success("Top 5 similar products:")
                for i, prod in enumerate(recommendations, 1):
                    st.write(f"{i}. {prod}")
            else:
                st.warning("Product not found. Please try a different name.")
        else:
            st.warning("Please select a product.")

# ---------------------------------
# Customer Segmentation Tab
# ---------------------------------
with tab2:
    st.header("Customer Segmentation Predictor")

    recency = st.number_input("Recency (in days)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, format="%.2f")

    if st.button("Predict Cluster"):
        if recency == 0 and frequency == 0 and monetary == 0.0:
            st.warning("Please enter valid RFM values before prediction.")
        else:
            input_data = np.array([[recency, frequency, monetary]])
            input_scaled = scaler.transform(input_data)
            cluster_label = kmeans.predict(input_scaled)[0]

            # Only 3 segments here, change labels as you want
            cluster_names = {
                0: "High-Value",
                1: "Regular",
                2: "Occasional"
            }

            if cluster_label not in cluster_names:
                st.error("Prediction outside expected clusters.")
            else:
                segment = cluster_names[cluster_label]
                st.success(f"The customer belongs to the **{segment} customer segment**.")

# Optional footer
st.markdown("---")
st.caption("Developed for Shopper Spectrum Capstone Project")
