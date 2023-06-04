import streamlit as st
import os
from fastai.vision.all import *

path = os.path.dirname(os.path.abspath(__file__)) #找到文件夹目录
model_path = os.path.join(path,'export.pkl')  #链接目录与文件

uploaded_file = st.file_uploader("choose an image...",type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img.to_thumb(500,500),caption='Your Image')
    pred,pred_idx,probs = learn_inf.predict(img)
    st.write(f'Prediction:{pred};Probability:{probs[pred_idx]:.04f}')

