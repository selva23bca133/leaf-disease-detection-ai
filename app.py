import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

st.set_page_config(
    page_title="Leaf Health Check",
    page_icon="🌿",
    layout="wide"
)

model = tf.keras.models.load_model("model.h5")

# sidebar
st.sidebar.title("🌿 Leaf Health Check")
page = st.sidebar.radio(
    "Navigation",
    ["Analyze Leaf", "About"]
)

# animated title
st.markdown(
    """
    <h1 style='text-align:center; color:#4CAF50;'>
    🌿 Leaf Health Check
    </h1>
    """,
    unsafe_allow_html=True
)

# main page
if page == "Analyze Leaf":

    st.subheader("Upload & Analyze Leaf Image")

    col1, col2 = st.columns(2)

    with col1:

        uploaded_file = st.file_uploader(
            "Upload leaf image",
            type=["jpg","png","jpeg"]
        )

        if uploaded_file is not None:

            img = Image.open(uploaded_file).convert("RGB")

            st.image(
                img,
                caption="Uploaded Leaf",
                use_container_width=True
            )

            img = img.resize((128,128))
            img = np.array(img)/255.0
            img = np.expand_dims(img,axis=0)

    with col2:

        st.write("### Analysis Result")

        if uploaded_file is None:

            st.info("Upload a leaf image to start analysis.")

        else:

            with st.spinner("🔍 AI analyzing leaf..."):
                time.sleep(2)

                pred = model.predict(img)[0][0]

            st.write("Model Confidence:", round(float(pred),3))

            if pred < 0.3:

                st.error("⚠ Diseased Leaf Detected")

                st.markdown("""
                ### Recommendation
                - Remove infected leaves
                - Use organic fungicide
                - Improve airflow around plant
                """)

            else:

                st.success("✅ Healthy Leaf")

                st.markdown("""
                ### Recommendation
                - Maintain watering schedule
                - Provide sunlight
                - Monitor plant health
                """)

# about page
elif page == "About":

    st.title("About Project")

    st.write("""
    AI system that detects plant leaf disease using CNN.

    Technologies:
    - Python
    - TensorFlow
    - Streamlit
    - Deep Learning
    """)