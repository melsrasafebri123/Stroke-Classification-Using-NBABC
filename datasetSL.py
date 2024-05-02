# Impor library yang diperlukan
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class preprocessing:
    def __init__(self,filename):
        self.filename = filename

    def preprocess(self):
        # Memuat dataset Stroke Prediction (ganti dengan jalur file dataset yang sesuai)
        datasetStroke = pd.read_csv(self.filename)

        #Menghapus kolom id karena tidak diperlukan
        datasetStroke.drop('id', axis=1, inplace=True)

        #Mengubah tipe data Age dari float menjadi int
        datasetStroke['age'] = datasetStroke['age'].astype(int)

        #Menghapus gender bernilai other karena tidak diperlukan dan hanya 1 baris
        datasetStroke = datasetStroke[datasetStroke['gender'] != 'Other']

        #Mengisi nilai rata-rata untuk value kosong pada kolom bmi
        datasetStroke['bmi'].fillna(datasetStroke['bmi'].mean(), inplace=True)

        #Label Encoder untuk kolom kategorikal
        label_encoder = LabelEncoder()
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for column in categorical_columns:
            datasetStroke[column] = label_encoder.fit_transform(datasetStroke[column])
            with open(f"{column}.pkl", "wb") as f:
                pickle.dump(label_encoder, f)
        
        # Memisahkan fitur dan target
        X = datasetStroke.drop(['stroke'], axis=1)  # Kolom 'stroke' adalah target
        y = datasetStroke['stroke']

        # Mengacak data
        X, y = shuffle(X, y,random_state=123)

        # Melakukan standard scaling pada fitur
        numeric_columns = ['avg_glucose_level', 'bmi']
        scaler = StandardScaler()
        # Menggunakan fit_transform untuk melakukan standarisasi pada kolom numerik
        datasetStroke[numeric_columns] = scaler.fit_transform(datasetStroke[numeric_columns])

        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # Memisahkan data menjadi set pelatihan dan pengujian
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        #SMOTE
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_test, y_test = smote.fit_resample(X_test, y_test)

        return X_train, X_test, y_train, y_test
    
    def preprocess_test(self):
        # Memuat dataset Stroke Prediction (ganti dengan jalur file dataset yang sesuai)
        datasetStroke = pd.read_csv(self.filename)

        #Menghapus kolom id karena tidak diperlukan
        datasetStroke.drop('id', axis=1, inplace=True)

        #Mengubah tipe data Age dari float menjadi int
        datasetStroke['age'] = datasetStroke['age'].astype(int)

        #Menghapus gender bernilai other karena tidak diperlukan dan hanya 1 baris
        datasetStroke = datasetStroke[datasetStroke['gender'] != 'Other']

        #Mengisi nilai rata-rata untuk value kosong pada kolom bmi
        datasetStroke['bmi'].fillna(datasetStroke['bmi'].mean(), inplace=True)

        #Label Encoder untuk kolom kategorikal
        label_encoder = LabelEncoder()
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status','age']
        for column in categorical_columns:
            with open(f"{column}.pkl", "wb") as f:
                pickle.dump(label_encoder, f)
            datasetStroke[column] = label_encoder.transform(datasetStroke[column])
        
        # Memisahkan fitur dan target
        X = datasetStroke.drop(['stroke'], axis=1)  # Kolom 'stroke' adalah target
        y = datasetStroke['stroke']

        # Melakukan standard scaling pada fitur
        with open("scaler.pkl", "rb") as f:
            scaler= pickle.load(f)
        X = scaler.transform(X)


        return X,y