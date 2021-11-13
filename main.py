import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model

model = load_model('new_model_vgg_16.h5',compile=False)
lab = {0: 'No DR',1:"Mild",2:"Moderate",3:"Severe",4:"Proliferative DR"}
#0 - No DR
# 1 - Mild
# 2 - Moderate
# 3 - Severe
# 4 - Proliferative DR

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3)) #256,256,3 for  my-model-dr-256
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def run():
    st.title("Indian Diabetic Retinopathy Image Classification")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "Indian Diabetic Retinopathy Image Dataset"</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Eye", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width='auto')
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Predicted DR grade is: "+result)
run()