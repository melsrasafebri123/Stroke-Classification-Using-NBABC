import streamlit as st
import pickle
import numpy as np
import pandas as pd
from NBSL import NaiveBayesClassifier
from sklearn.naive_bayes import GaussianNB
from datasetSL import preprocessing
from ABCSL import FeatureSelectionABC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class app:
    st.set_page_config(page_title="NB + ABC")

    # Function to load the trained model
    def load_model_and_accuracy(model_path, accuracy_path, feature_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(accuracy_path, 'rb') as file:
            accuracy = pickle.load(file)
        with open(feature_path, 'rb') as file:
            feature_selected = pickle.load(file)
        return model, accuracy, feature_selected

    # Initialize the classifier
    nb_classifier = NaiveBayesClassifier()
    data = preprocessing(filename='')

    # File uploader for CSV
    with st.form(key='classification_form'):
        st.subheader(
            'OPTIMASI ALGORITMA NAÏVE BAYES MENGGUNAKAN ARTIFICIAL BEE COLONY UNTUK KLASIFIKASI DATA PENDERITA PENYAKIT STROKE')
        st.write('Melsra Safebri - 09021282025055 - Teknik Informatika 2020')
        st.markdown("""
            <style>
                .separator {
                    margin-top: -10px;
                    border-top: 3px solid black;
                }
            </style>
        """, unsafe_allow_html=True)

        # Menampilkan garis pemisah
        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader('Upload dataset (.csv)', type='csv')

        num_bees = st.radio('Jumlah Lebah:', [10, 20, 30, 40, 50])
        st.write(
            '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        max_iter = st.radio('Jumlah Iterasi:', [25, 50, 75, 100])
        st.write(
            '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        submit_button = st.form_submit_button(label='Mulai')

        if submit_button:
            if uploaded_file is not None:
                data = preprocessing(uploaded_file)
                X_train, X_test, y_train, y_test = data.preprocess()

                # Train the model with the training set
                nb_model = GaussianNB()
                nb_model.fit(X_train, y_train)

                # Save the trained model and its accuracy
                with open('model1.pkl', 'wb') as f:
                    pickle.dump(nb_model, f)

                col1, col2 = st.columns(2)

                with col1:
                    y_pred = nb_model.predict(X_test)
                    accuracy_NB = accuracy_score(y_test, y_pred)
                    precision_NB = precision_score(y_test, y_pred)
                    recall_NB = recall_score(y_test, y_pred)
                    f1score_NB = f1_score(y_test, y_pred)

                with col2:
                    feature_selector = FeatureSelectionABC(
                        num_bees, max_iter, position=None)
                    best_feature_mask_position, _ = feature_selector.optimize_bees(
                        X_train, X_test, y_train, y_test)

                    classifier = NaiveBayesClassifier()
                    nb_model_after_selection = classifier.train(
                        X_train, y_train, best_feature_mask_position)
                    y_pred_after_selection = classifier.test(
                        nb_model_after_selection, X_test, best_feature_mask_position)
                    accuracy_after_selection = accuracy_score(
                        y_test, y_pred_after_selection)
                    precision_NBABC = precision_score(
                        y_test, y_pred_after_selection)
                    recall_NBABC = recall_score(y_test, y_pred_after_selection)
                    f1score_NBABC = f1_score(y_test, y_pred_after_selection)

                    # Save the trained model and its accuracy, precision, recall, f1 score
                    with open('model2.pkl', 'wb') as f:
                        pickle.dump(nb_model_after_selection, f)
                    with open('model_accuracy.pkl', 'wb') as f:
                        pickle.dump(accuracy_after_selection, f)
                    with open('feature.pkl', 'wb') as f:
                        pickle.dump(best_feature_mask_position, f)

                feature_idx = np.where(best_feature_mask_position == 1)[0]
                feature = X_train.columns[best_feature_mask_position == 1].tolist(
                )

                data = {
                    " ": ["Akurasi", "Precision", "Recall", "F1score", "Index Fitur", "Fitur"],
                    "Naive Bayes": [accuracy_NB, precision_NB, recall_NB, f1score_NB, "-", "-"],
                    "Naive Bayes + Artificial Bee Colony": [accuracy_after_selection, precision_NBABC,
                                                            recall_NBABC, f1score_NBABC, feature_idx, feature]
                }

                # Membuat DataFrame dari dictionary
                df = pd.DataFrame(data)

                # Mengonversi DataFrame ke tabel HTML tanpa indeks
                html = df.to_html(index=False)

                # Menambahkan style CSS untuk memposisikan nama kolom di tengah
                html = html.replace('<th>', '<th style="text-align:center;">')
                html = html.replace(
                    '<tr>', '<tr style="text-align:center; vertical-align: middle;">')
                html = html.replace('<table border="1" class="dataframe">',
                                    '<table border="1" class="dataframe" style="width:100%">')

                # Menampilkan DataFrame di Streamlit
                st.write(html, unsafe_allow_html=True)
                st.write(" ")

            else:
                st.warning('Upload File Dahulu!', icon="⚠️")
