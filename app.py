
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO("/content/runs/detect/train/weights/best.pt")

model = load_model()

st.title("🤖 YOLO Object Detection System")
st.write("Upload an image and detect objects using my trained YOLO model.")

st.divider()

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        image.save(temp_file.name)
        temp_path = temp_file.name

    if st.button("🔍 Detect Objects"):

        results = model.predict(
            source=temp_path,
            conf=0.25
        )

        result = results[0]

        result_image = result.plot()

        with col2:
            st.subheader("Detection Result")

            st.image(
                result_image,
                channels="BGR",
                use_container_width=True
            )

        st.subheader("📊 Detection Information")

        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:

            for box in boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                st.write(
                    f"**{class_name}** — "
                    f"Confidence: **{confidence:.2%}**"
                )

        else:
            st.warning("No objects detected.")

    os.remove(temp_path)
