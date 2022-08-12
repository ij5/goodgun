from io import BytesIO
import streamlit as st
import main

f = st.file_uploader('Source', ['png', 'jpg', 'jpeg'], False, help=".png, .jpeg, .jpg 파일만 지원됩니다.")

if f is not None:
    img = BytesIO(f.read())
    img.seek(0)
    resultbytes = main.generate(img)
    result = BytesIO(resultbytes)
    result.seek(0)
    st.image(result, caption="Generated Image")