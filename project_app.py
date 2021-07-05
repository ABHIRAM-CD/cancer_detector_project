import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import io



st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)

def load_model():
  model = tf.keras.models.load_model('SERS_model.hdf5')
  return model

model = load_model()

st.title("SERS Data Cancer Cell Detector")
st.write("Each row of the file is treated as a single spectra. (Upload all three spectra files for better accuracy)")

#taking input from user and processing as input

spectra_1_type = st.selectbox('Select Fuctional group for first spectra:',['Nil', 'COOH2', 'COOH', 'NH2'], key = "spectra_1")

if spectra_1_type != 'Nil':
    spectra_1_label = "Upload the file for spctrum with " + spectra_1_type + " functional group"
    spectra_1_uploaded = st.file_uploader(spectra_1_label, key = "spectra_1_file")
    if spectra_1_uploaded is not None:
        spectra_1 = pd.read_csv(spectra_1_uploaded)

spectra_2_type = st.selectbox('Select Fuctional group for second spectra:',['Nil', 'COOH2', 'COOH', 'NH2'], key = "spectra_2")

if spectra_2_type != 'Nil':
    spectra_2_label = "Upload the file for spctrum with " + spectra_2_type + " functional group"
    spectra_2_uploaded = st.file_uploader(spectra_2_label, key = "spectra_2_file")
    if spectra_2_uploaded is not None:
        spectra_2 = pd.read_csv(spectra_2_uploaded)

spectra_3_type = st.selectbox('Select Fuctional group for third spectra:',['Nil', 'COOH2', 'COOH', 'NH2'], key = "spectra_3")

if spectra_3_type != 'Nil':
    spectra_3_label = "Upload the file for spctrum with " + spectra_3_type + " functional group"
    spectra_3_uploaded = st.file_uploader(spectra_3_label, key = "spectra_3_file")
    if spectra_3_uploaded is not None:
        spectra_3 = pd.read_csv(spectra_3_uploaded)

#prediction functions for different number of inputs

def prediction_single(spectra_1, model):
    rows=spectra_1.shape[0]
    cols=spectra_1.shape[1]

    spectra_1=spectra_1.iloc[:, 0:2090]

    spectra_zero = pd.DataFrame(np.zeros((rows,cols)))

    spectra_1 = tf.reshape(spectra_1, [ rows, cols, 1])
    spectra_zero = tf.reshape(spectra_1, [ rows, cols, 1])

    if spectra_1_type == 'COOH2':
            prediction = model.predict([spectra_1,spectra_zero,spectra_zero])
    elif spectra_1_type == 'COOH':
            prediction = model.predict([spectra_zero,spectra_1,spectra_zero])
    else:
            prediction = model.predict([spectra_zero,spectra_zero,spectra_1])

    return prediction

#output display for each row

if st.button('Predict Chances'):
    if spectra_1_type is not 'Nil':
            probability = prediction_single(spectra_1, model)

            for i in range(len(probability)):
                if probability[i]<0.3:
                    p = round(probability[i][0],3)
                    label = "The sample (" + str(i+1) + ") is more like to be a Normal cell with probability of cancer : " + str(p)
                    string_out = label
                    st.success(string_out)

                elif probability[i]>0.7:
                    p = round(probability[i][0],3)
                    label = "The sample (" + str(i+1) + ") is more like to be a Cancer cell with probability of cancer : " + str(p)
                    string_out = label
                    st.error(string_out)

                else:
                    p = round(probability[i][0],3)
                    label = "The sample (" + str(i+1) + ") is having a probability of cancer : " + str(p)
                    string_out = label
                    st.warning(string_out)

#end NOTE:

st.markdown("""
<style>
.big-font {
    font-size:12px;
    padding:20px;
    text-align:center;
    color:#AAA;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Sixth Semester project of Abhiram CD and Parthana Jayaprakash<br>International School of Photonics Cochin University of Science and Technology Cochin-22, India</p>', unsafe_allow_html=True)
