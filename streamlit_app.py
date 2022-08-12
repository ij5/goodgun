from io import BytesIO
import streamlit as st
import main

st.markdown("""
# 굳건이 생성기
캐릭터의 얼굴을 인식하여 굳건이 얼굴로 변환합니다.
2D 캐릭터 얼굴을 인풋으로 넣었을 때 가장 잘 작동합니다.
""")

f = st.file_uploader('Input Image', ['png', 'jpg', 'jpeg'], False, help=".png, .jpeg, .jpg 파일만 지원됩니다.")

if f is not None:
    img = BytesIO(f.read())
    img.seek(0)
    resultbytes = main.generate(img)
    result = BytesIO(resultbytes)
    result.seek(0)
    st.image(result, caption="Generated Image")

st.markdown("by [이재희](https://github.com/ij5)")