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

st.markdown("""
<style>
.foot {
    font-size:12px;
    padding:20px;
    text-align:center;
    color:#AAA;
}
.notes{
    font-size:15px;
    color:#AAA;
    padding:0px;
    margin:0px;
}
.hr{
    border: 1px solid #CCC;
    border-radius: 10px;
    margin:5px;
}
</style>
""", unsafe_allow_html=True)

st.title("Cancer Cell Detector from SERS Data")

st.markdown('<hr class="hr">', unsafe_allow_html=True)

st.write("**About The Application**")
st.markdown('<p class = "notes" style="color:#000; padding-top:10px; padding-bottom:10px;">Cancer Cell Detector from SERS Data is a web application for detecting the probability of a sample is a cancer cell or a normal cell directly by examining the surface enhanced raman spectra valus of the sample.</p>', unsafe_allow_html=True)
st.markdown('<p class = "notes">Each row of the file is treated as a single spectra. (Upload all three spectra files for better accuracy)</p>', unsafe_allow_html=True)
st.markdown('<p class = "notes" style="padding-bottom: 10px;">note : Each corresponding rows in each files are processed together</p>', unsafe_allow_html=True)

st.markdown('<hr class="hr">', unsafe_allow_html=True)

st.write('**Predictions**')

#taking input from user and processing as input

spectra_1_type = st.selectbox('Select Fuctional group for first spectra:',['Nil', 'COOH2', 'COOH', 'NH2'], key = "spectra_1")

if spectra_1_type != 'Nil':
    spectra_1_label = "Upload the file for spctrum with " + spectra_1_type + " functional group"
    spectra_1_uploaded = st.file_uploader(spectra_1_label, key = "spectra_1_file")
    if spectra_1_uploaded is not None:
        spectra_1_file = pd.read_csv(spectra_1_uploaded)


st.markdown('<hr class="hr">', unsafe_allow_html=True)


spectra_2_type = st.selectbox('Select Fuctional group for second spectra:',['Nil', 'COOH2', 'COOH', 'NH2'], key = "spectra_2")

if spectra_2_type == spectra_1_type and spectra_2_type != 'Nil':
    st.write("Already selcted option")
else:
    if spectra_2_type != 'Nil':
        spectra_2_label = "Upload the file for spctrum with " + spectra_2_type + " functional group"
        spectra_2_uploaded = st.file_uploader(spectra_2_label, key = "spectra_2_file")
        if spectra_2_uploaded is not None:
            spectra_2_file = pd.read_csv(spectra_2_uploaded)


st.markdown('<hr class="hr">', unsafe_allow_html=True)


spectra_3_type = st.selectbox('Select Fuctional group for third spectra:',['Nil', 'COOH2', 'COOH', 'NH2'], key = "spectra_3")

if (spectra_3_type == spectra_1_type or spectra_3_type == spectra_2_type) and spectra_2_type != 'Nil':
    st.write("Already selcted option")
else:
    if spectra_3_type != 'Nil':
        spectra_3_label = "Upload the file for spctrum with " + spectra_3_type + " functional group"
        spectra_3_uploaded = st.file_uploader(spectra_3_label, key = "spectra_3_file")
        if spectra_3_uploaded is not None:
            spectra_3_file = pd.read_csv(spectra_3_uploaded)


st.markdown('<hr class="hr">', unsafe_allow_html=True)

#prediction functions for different number of inputs

def prediction_single(spectra_1, model, type_1):
    rows=spectra_1.shape[0]
    cols=spectra_1.shape[1]

    if cols >= 2090:
        middle_1 = int(cols/2)
        spectra_1_inp = spectra_1.iloc[:, middle_1-1045:middle_1+1045]
    else:
        diff_1 = 2090-cols
        add_1 = pd.DataFrame(np.zeros((rows, diff_1)))
        spectra_1_inp = pd.concat([spctra_1, add_1], axis = 1)

    spectra_zero = pd.DataFrame(np.zeros((rows, 2090)))

    spectra_1_pred = tf.reshape(spectra_1, [ rows, 2090, 1])
    spectra_zero = tf.reshape(spectra_zero, [ rows, 2090, 1])

    if type_1 == 'COOH2':
        prediction = model.predict([spectra_1_pred, spectra_zero, spectra_zero])
    elif type_1 == 'COOH':
        prediction = model.predict([spectra_zero, spectra_1_pred, spectra_zero])
    else:
        prediction = model.predict([spectra_zero, spectra_zero, spectra_1_pred])

    return prediction

def prediction_double(spectra_1, spectra_2, model, type_1, type_2):
    rows_1=spectra_1.shape[0]
    cols_1=spectra_1.shape[1]

    rows_2=spectra_2.shape[0]
    cols_2=spectra_2.shape[1]

    if cols_1 >= 2090:
        middle_1 = int(cols_1/2)
        spectra_1 = spectra_1.iloc[:, middle_1-1045:middle_1+1045]
    else:
        diff_1 = 2090-cols_1
        add_1 = np.zeros((rows_1, diff_1))
        spectra_1 = np.concatenate([np.array(spctra_1), add_1], axis = 1)

    if cols_2 >= 2090:
        middle_2 = int(cols_2/2)
        spectra_2 = spectra_2.iloc[:, middle_2-1045:middle_2+1045]
    else:
        diff_2 = 2090-cols_2
        add_2 = np.zeros((rows_2, diff_2))
        spectra_2 = np.concatenate([np.array(spctra_2), add_2], axis = 1)

    if rows_1 != rows_2:
        diff = abs(rows_1-rows_2)
        add = np.zeros((diff, 2090))
        if rows_1 < rows_2:
            spectra_1_inp = np.concatenate([spectra_1, add], axis = 0)
            spectra_2_inp = spectra_2
            rows = rows_2
        else:
            spectra_2_inp = np.concatenate([spectra_2, add], axis = 0)
            spectra_1_inp = spectra_1
            rows = rows_1
        st.write("Rows mismatch! zero values added for some rows.")
    else:
        rows = rows_1
        spectra_1_inp = spectra_1
        spectra_2_inp = spectra_2

    spectra_zero = np.zeros((rows, 2090))

    spectra_1_pred = tf.reshape(spectra_1_inp, [ rows, 2090, 1])
    spectra_2_pred = tf.reshape(spectra_2_inp, [ rows, 2090, 1])
    spectra_zero = tf.reshape(spectra_zero, [ rows, 2090, 1])

    if type_1 == 'COOH2':
        if type_2 == 'COOH':
            prediction = model.predict([spectra_1_pred, spectra_2_pred, spectra_zero])
        else:
            prediction = model.predict([spectra_1_pred, spectra_zero, spectra_2_pred])
    elif type_1 == 'COOH':
        if type_2 == 'COOH2':
            prediction = model.predict([spectra_2_pred, spectra_1_pred, spectra_zero])
        else:
            prediction = model.predict([spectra_zero, spectra_1_pred, spectra_2_pred])
    else:
        if type_2 == 'COOH2':
            prediction = model.predict([spectra_2_pred, spectra_zero, spectra_1_pred])
        else:
            prediction = model.predict([spectra_zero, spectra_2_pred, spectra_1_pred])

    return prediction

def prediction_triple(spectra_1, spectra_2, spectra_3, model, type_1, type_2, type_3):
    rows_1=spectra_1.shape[0]
    cols_1=spectra_1.shape[1]

    rows_2=spectra_2.shape[0]
    cols_2=spectra_2.shape[1]

    rows_3=spectra_3.shape[0]
    cols_3=spectra_3.shape[1]

    if cols_1 >= 2090:
        middle_1 = int(cols_1/2)
        spectra_1 = spectra_1.iloc[:, middle_1-1045:middle_1+1045]
    else:
        diff_1 = 2090-cols_1
        add_1 = np.zeros((rows_1, diff_1))
        spectra_1 = np.concatenate([np.array(spctra_1), add_1], axis = 1)

    if cols_2 >= 2090:
        middle_2 = int(cols_2/2)
        spectra_2 = spectra_2.iloc[:, middle_2-1045:middle_2+1045]
    else:
        diff_2 = 2090-cols_2
        add_2 = np.zeros((rows_2, diff_2))
        spectra_2 = np.concatenate([np.array(spctra_2), add_2], axis = 1)

    if cols_3 >= 2090:
        middle_3 = int(cols_3/2)
        spectra_3 = spectra_3.iloc[:, middle_3-1045:middle_3+1045]
    else:
        diff_3 = 2090-cols_3
        add_3 = np.zeros((rows_3, diff_3))
        spectra_3 = np.concatenate([np.array(spctra_3), add_3], axis = 1)

    if rows_1 != rows_2 or rows_1 != rows_3 or rows_2 != rows_3:

        spectra_1_inp = spectra_1
        spectra_2_inp = spectra_2
        spectra_3_inp = spectra_3

        if rows_1 != rows_2:
            diff = abs(rows_1-rows_2)
            add = np.zeros((diff, 2090))
            if rows_1 < rows_2:
                spectra_1_inp = np.concatenate([spectra_1_inp, add], axis = 0)
            else:
                spectra_2_inp = np.concatenate([spectra_2_inp, add], axis = 0)

        rows_1=spectra_1_inp.shape[0]
        rows_3=spectra_3_inp.shape[0]

        if rows_1 != rows_3:
            diff = abs(rows_1-rows_3)
            add = np.zeros((diff, 2090))
            if rows_1 < rows_3:
                spectra_1_inp = np.concatenate([spectra_1_inp, add], axis = 0)
            else:
                spectra_3_inp = np.concatenate([spectra_3_inp, add], axis = 0)

        rows_2=spectra_2_inp.shape[0]
        rows_3=spectra_3_inp.shape[0]

        if rows_3 != rows_2:
            diff = abs(rows_3-rows_2)
            add = np.zeros((diff, 2090))
            if rows_3 < rows_2:
                spectra_3_inp = np.concatenate([spectra_3_inp, add], axis = 0)
            else:
                spectra_2_inp = np.concatenate([spectra_2_inp, add], axis = 0)

        rows = max(rows_1, rows_2, rows_3)
        st.write("Rows mismatch! zero values added for some rows.")
    else:
        rows = rows_1
        spectra_1_inp = spectra_1
        spectra_2_inp = spectra_2
        spectra_3_inp = spectra_3

    spectra_zero = np.zeros((rows, 2090))

    spectra_1_pred = tf.reshape(spectra_1_inp, [ rows, 2090, 1])
    spectra_2_pred = tf.reshape(spectra_2_inp, [ rows, 2090, 1])
    spectra_3_pred = tf.reshape(spectra_3_inp, [ rows, 2090, 1])
    spectra_zero = tf.reshape(spectra_zero, [ rows, 2090, 1])

    if type_1 == 'COOH2':
        if type_2 == 'COOH':
            prediction = model.predict([spectra_1_pred, spectra_2_pred, spectra_3_pred])
        else:
            prediction = model.predict([spectra_1_pred, spectra_3_pred, spectra_2_pred])
    elif type_1 == 'COOH':
        if type_2 == 'COOH2':
            prediction = model.predict([spectra_2_pred, spectra_1_pred, spectra_3_pred])
        else:
            prediction = model.predict([spectra_3_pred, spectra_1_pred, spectra_2_pred])
    else:
        if type_2 == 'COOH2':
            prediction = model.predict([spectra_2_pred, spectra_3_pred, spectra_1_pred])
        else:
            prediction = model.predict([spectra_3_pred, spectra_2_pred, spectra_1_pred])

    return prediction


#output display for each row

if st.button('Predict Chances'):

    if spectra_1_type is 'Nil':
        s1 = 0
    else:
        s1 = 1

    if spectra_2_type is 'Nil':
        s2 = 0
    else:
        s2 = 2

    if spectra_3_type is 'Nil':
        s3 = 0
    else:
        s3 = 4

    if s1+s2+s3 == 0:
        st.error("No options selected!")
        probability = [10]

    if s1+s2+s3 == 1:
        probability = prediction_single(spectra_1_file, model, spectra_1_type)
    elif s1+s2+s3 == 2:
        probability = prediction_single(spectra_2_file, model, spectra_2_type)
    elif s1+s2+s3 == 4:
        probability = prediction_single(spectra_3_file, model, spectra_3_type)
    elif s1+s2+s3 == 3:
        probability = prediction_double(spectra_1_file, spectra_2_file, model, spectra_1_type, spectra_2_type)
    elif s1+s2+s3 == 5:
        probability = prediction_double(spectra_1_file, spectra_3_file, model, spectra_1_type, spectra_3_type)
    elif s1+s2+s3 == 6:
        probability = prediction_double(spectra_2_file, spectra_3_file, model, spectra_2_type, spectra_3_type)
    elif s1+s2+s3 == 7:
        probability = prediction_triple(spectra_1_file, spectra_2_file, spectra_3_file, model, spectra_1_type, spectra_2_type, spectra_3_type)

    for i in range(len(probability)):

        if probability[i] == 10:
            break

        elif probability[i]<0.3:
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

st.markdown('<hr class="hr" style="margin-top:50%; margin-bottom: 5px;">', unsafe_allow_html=True)

st.markdown('<p class="foot">Sixth Semester project of Abhiram CD and Parthana Jayaprakash<br>International School of Photonics Cochin University of Science and Technology Cochin-22, India</p>', unsafe_allow_html=True)
