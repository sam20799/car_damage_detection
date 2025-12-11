import streamlit as st
from model_helper import predict
st.title("Vehicle Damage Detector")

uploaded_file = st.file_uploader("Upload the file",type=['png','jpeg'])

if uploaded_file:
    img_path = 'temp_file.jpeg'
    with open(img_path,'wb') as f:
        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file,caption='Uploaded File',use_container_width=True)
        prediction = predict(img_path)
        st.info(f'Predicted Class: {prediction}')
        print(prediction)


