import streamlit as st
import pickle
import numpy as np
import pandas as pd
from NBSL import NaiveBayesClassifier
from sklearn.naive_bayes import GaussianNB
from datasetSL import preprocessing
from ABCSL import FeatureSelectionABC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from streamlit_option_menu import option_menu

class app:
        st.set_page_config(page_title="NB + ABC")

        # Function to load the trained model
        def load_model_and_accuracy(model_path, accuracy_path,feature_path):
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
            st.subheader('OPTIMASI ALGORITMA NAÏVE BAYES MENGGUNAKAN ARTIFICIAL BEE COLONY UNTUK KLASIFIKASI DATA PENDERITA PENYAKIT STROKE')
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

            col1, col2 = st.columns(2)
            with col1:
                num_bees = st.radio('Jumlah Lebah:', [10,20,30,40,50])
                st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

            with col2:
                max_iter = st.radio('Jumlah Iterasi:', [25,50,75,100])
                st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

            submit_button = st.form_submit_button(label='Mulai')

            if submit_button:
                if uploaded_file is not None:
                    data = preprocessing(uploaded_file)
                    X_train, X_test, y_train, y_test = data.preprocess()

                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # st.header("Naive Bayes")
                        # Train the model with the training set
                        nb_model = GaussianNB()
                        nb_model.fit(X_train, y_train)
                        y_pred = nb_model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1score = f1_score(y_test, y_pred)

                        # Save the trained model and its accuracy
                        with open('model1.pkl', 'wb') as f:
                            pickle.dump(nb_model, f)
                        with open('model_accuracy.pkl', 'wb') as f:
                            pickle.dump(accuracy, f)
                        # st.write('Model trained and saved successfully!')
                        accuracy_NB = accuracy
                        precision_NB = precision
                        recall_NB = recall
                        f1score_NB = f1score

                    with col2:
                        # Inisialisasi objek FeatureSelectionABC dengan parameter yang sesuai
                        feature_selector = FeatureSelectionABC( num_bees, max_iter, position=None)
                        best_feature_mask_position, best_feature_mask_score = feature_selector.optimize_bees(X_train, X_test, y_train, y_test)
                        
                        # Training dan Testing Akurasi Model Naive Bayes setelah seleksi fitur
                        classifier = NaiveBayesClassifier()
                        nb_model_after_selection = classifier.train(X_train, y_train, best_feature_mask_position)
                        y_pred_after_selection = classifier.test(nb_model_after_selection, X_test, best_feature_mask_position)
                        accuracy_after_selection = accuracy_score(y_test, y_pred_after_selection)
                        precision = precision_score(y_test, y_pred_after_selection)
                        recall = recall_score(y_test, y_pred_after_selection)
                        f1score = f1_score(y_test, y_pred_after_selection)

                        # Save the trained model and its accuracy, precision, recall, f1 score
                        with open('model2.pkl', 'wb') as f:
                            pickle.dump(nb_model_after_selection, f)
                        with open('model_accuracy.pkl', 'wb') as f:
                            pickle.dump(accuracy_after_selection, f)
                        with open('feature.pkl', 'wb') as f:
                            pickle.dump(best_feature_mask_position, f)

                        # st.write('Model trained and saved successfully!')

                        accuracy_NBABC = accuracy_after_selection
                        precision_NBABC = precision
                        recall_NBABC = recall
                        f1score_NBABC = f1score
                        feature_idx = np.where(best_feature_mask_position==1)[0]
                        feature = X_train.columns[best_feature_mask_position == 1].tolist()

                    data = {
                        " ": ["Akurasi","Precision", "Recall", "F1score","Index Fitur", "Fitur"],
                        "Naive Bayes": [accuracy_NB, precision_NB, recall_NB, f1score_NB,"-","-"],
                        "Naive Bayes + Artificial Bee Colony": [accuracy_NBABC, precision_NBABC, recall_NBABC, f1score_NBABC,feature_idx,feature]
                    }

                    # Membuat DataFrame dari dictionary
                    df = pd.DataFrame(data)

                    # Mengonversi DataFrame ke tabel HTML tanpa indeks
                    html = df.to_html(index=False)

                    # Menambahkan style CSS untuk memposisikan nama kolom di tengah
                    html = html.replace('<th>', '<th style="text-align:center;">')
                    html = html.replace('<tr>', '<tr style="text-align:center; vertical-align: middle;">')
                    html = html.replace('<table border="1" class="dataframe">', '<table border="1" class="dataframe" style="width:100%">')

                    # Menampilkan DataFrame di Streamlit
                    st.write(html, unsafe_allow_html=True)
                    st.write(" ")

                    #Confusion Matrix
                    # conf_matrix = confusion_matrix(y_test, y_pred)
                    # conf_matrix = confusion_matrix(y_test, y_pred_after_selection)
                    # print(conf_matrix)
                    # print(best_feature_mask_position)
                    # print(best_feature_mask_score)
                else:
                    st.warning('Upload File Dahulu!', icon="⚠️") 