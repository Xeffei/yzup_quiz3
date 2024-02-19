import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score, f1_score,confusion_matrix,recall_score
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.naive_bayes import GaussianNB



class App:
    def __init__(self):
        self.veri = pd.read_csv("data.csv")
        self.name = "data.csv"
        self.dataset_name = None
        self.metod = ""
    
    def run(self):
        self.dosya_al()
        st.write(f"### Çalıştırılan Dosya: {self.name}")
        st.write("## Yüklenen Dosyanın son 10 satırı")
        self.get_dataset()
        self.Init_Streamlit_Page()
        self.grafik()
        self.get_ideal()
        self.korelasyon_m()
        self.matrx()
        
               
    def Init_Streamlit_Page(self):
        self.metod = st.sidebar.selectbox("Select Method", ("KNN","SVM","Naive Bayes"))

    def get_dataset(self):
        self.X = self.veri.iloc[:,2:]
        self.y = self.veri.iloc[:,1:2].replace({'B': 1, 'M': 0})
        "st.write(self.X)"
        st.write(self.veri.tail(10))
        self.data_split()
               
    def grafik(self):
        
        st.subheader('Radius Mean vs Texture Mean')
        fig, ax = plt.subplots()
        sns.scatterplot(x='radius_mean', y='texture_mean', data=self.veri, hue='diagnosis', palette={'M': 'red', 'B': 'blue'}, ax=ax)
        st.pyplot(fig)
        
    def data_split(self):
        self.xtrain,self.xtest,self.ytrain,self.ytest = train_test_split(self.X,self.y,train_size=0.8,random_state=42)
          
    def get_ideal(self):
        if self.metod == "KNN":
            self.knn_ideal()
        elif self.metod == "SVM":    
            self.svm_ideal() 
        else:
            self.naive_bayes()
                            
    def knn_ideal(self):
        self.knn_param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2]}    
        knn_grid_search = GridSearchCV(KNeighborsClassifier(), self.knn_param_grid, cv=5, scoring='accuracy')
        knn_grid_search.fit(self.xtrain, self.ytrain)
        self.best_knn_params = knn_grid_search.best_params_
        self.knn()

    def svm_ideal(self):
        st.subheader('SVM Model Optimization')
        svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}

        with st.spinner('Optimizing SVM Model...'):
            svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=5, scoring='accuracy')
            svm_grid_search.fit(self.xtrain, self.ytrain)

        self.best_svm_params = svm_grid_search.best_params_
        self.best_svm_model = svm_grid_search.best_estimator_
        with st.spinner('Making Predictions...'): 
            self.svm()
      
    def naive_bayes(self):
        nai = GaussianNB()
        nai.fit(self.xtrain,self.ytrain)
        self.predict = nai.predict(self.xtest)
        self.acc = accuracy_score(self.ytest, self.predict)
        self.prec = precision_score(self.ytest, self.predict)
        self.f1 = f1_score(self.ytest, self.predict)
        self.rec = recall_score(self.ytest, self.predict)
                       
    def knn(self):
        knc = KNeighborsClassifier(n_neighbors= self.best_knn_params["n_neighbors"],p=self.best_knn_params["p"],weights=self.best_knn_params["weights"])
        knc.fit(self.xtrain,self.ytrain)
        self.predict = knc.predict(self.xtest)
        self.acc = accuracy_score(self.ytest, self.predict)
        self.prec = precision_score(self.ytest, self.predict)
        self.f1 = f1_score(self.ytest, self.predict)
        self.rec = recall_score(self.ytest, self.predict)
        
    def svm(self):
        svmm = svm.SVC(C= self.best_svm_params["C"],gamma=self.best_svm_params["gamma"],kernel=self.best_svm_params["kernel"])
        svmm.fit(self.xtrain,self.ytrain)
        self.predict = svmm.predict(self.xtest)
        self.acc = accuracy_score(self.ytest, self.predict)
        self.prec = precision_score(self.ytest, self.predict)
        self.f1 = f1_score(self.ytest, self.predict)
        self.rec = recall_score(self.ytest, self.predict)

    def matrx(self):
        
        cm = confusion_matrix(self.ytest, self.predict)
        st.write("## Confusion Matrix:")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.ylabel('Gerçek Sınıf')
        plt.title('Confusion Matrix')
        st.pyplot(plt)
        st.write("accuary: ",self.acc," precision: ",self.prec," recall: ",self.rec ," f1: ",self.f1)
    
    def dosya_al(self):
        uploaded_file = st.sidebar.file_uploader("CSV dosyasını seçin", type=["csv"])    

        if uploaded_file is not None:
            self.name = uploaded_file.name
            self.veri = pd.read_csv(uploaded_file)

    def korelasyon_m(self):
        st.subheader('Korelasyon Matrisi')
        malignant_data = self.veri[(self.veri['diagnosis'] == "M").replace(1)]
        benign_data = self.veri[(self.veri['diagnosis'] == "B").replace(0)]
        fig, ax = plt.subplots(1, 2, figsize=(20, 12))
        sns.heatmap(malignant_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax[0])
        ax[0].set_title('Malignant')
        sns.heatmap(benign_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax[1])
        ax[1].set_title('Benign')
        st.pyplot(fig)
