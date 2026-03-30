import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Annotation Fracture", layout="wide")
st.title("Annotation de fracture - Dessinez directement sur l'image")

# Upload image
uploaded_file = st.file_uploader("Choisissez une radiographie", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Radiographie originale", use_column_width=True)

    # Canvas pour annotation
    st.subheader("Dessinez un rectangle bleu sur la fracture")
    canvas_result = st_canvas(
        fill_color="rgba(0,0,255,0.3)",  # bleu transparent
        stroke_width=3,
        stroke_color="blue",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="rect",
        key="canvas"
    )

    # Affichage image annotée
    if canvas_result.image_data is not None:
        annotated_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGB')
        st.image(annotated_image, caption="Image annotée", use_column_width=True)
