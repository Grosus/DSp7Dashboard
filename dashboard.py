# Import des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import seaborn as sns
import os
from sklearn import metrics
from sklearn.metrics import recall_score, roc_auc_score, PrecisionRecallDisplay, RocCurveDisplay, fbeta_score,precision_score
import joblib
import shap
import pickle
from io import BytesIO
import requests
import json
st.set_option('deprecation.showPyplotGlobalUse', False)



# Lit le fichier nécéssaire au dashboard
@st.cache
def read(): 
    path= "./" 
    data=pd.read_csv(os.path.join(path,'df_prepro.csv'))
    df = data[data['TARGET'].notnull()].drop('Unnamed: 0',1)
    results=pd.read_csv(os.path.join(path,'prediction.csv'))
    loaded_model = joblib.load('model_lgbm.pkl')
    results['TARGET']=results.TARGET.round(2)
    with open("shap_value", "rb") as fp:   # Unpickling
        shap_values = pickle.load(fp)   
    expected_values =np.load('explainer.npy')
    return df , results , loaded_model,shap_values,expected_values  

# Plot de la partie client pour un client et une variable
def plot(df,options,number ):
    value_client=df.query('SK_ID_CURR == @number')[options]
    ymax=df.groupby(options).count().max()[0]
    if len(df[options].unique())<20:
        st.write('Valeurs de la variable pour le client :' ,value_client )
        fig, ax = plt.subplots()
        df_0=df[df['TARGET']==0]
        df_1=df[df['TARGET']==1]
        d = {'0': df_0[options].value_counts().sort_index(axis=0),
             '1': df_1[options].value_counts().sort_index(axis=0),
             }
        cat_plot= pd.DataFrame(data=d)
        cat_plot.plot(kind='bar',ax=ax)
        value_client=df.query('SK_ID_CURR == @number')[options]
        ax.plot(value_client,1 , marker = '|', linestyle = '' , color='green')
        plt.vlines(value_client,0,ymax, color='r', label='client')
    else:
        st.write('Valeurs de la variable pour le client :' ,value_client)
        fig=sns.displot(x=options,data=df,hue='TARGET')
        plt.vlines(value_client,0,ymax, color='r', label='client')
    
    return fig
    
#Selection de variable pour la fonction plot   
def client_plot(df,number):
    columns=list(df.drop(['SK_ID_CURR','TARGET'],1).columns)
    columns.insert(0,'Aucun')
    option = st.selectbox(
            'choix de la variable',
            columns)

    if option=='Aucun' :
        st.write('Choisissez une variable')
        st.stop()
    fig = plot(df , option,number)
    st.pyplot(fig)
    
#Fonction général de la partie client
def explo_plot(df,clf,shap_values,expected_values):
    st.title('Information sur les clients')
    st.write("Veuillez inserer un numero de client valide pour continuer l'éxploration")
    st.write("Comme ce Dashboard ne visualise qu'un nombre restreint de donnée ,des exemple d'ID clients valide peuvent etre trouvé dans la colonne SK_ID_CURR du fichier df_prepro.csv")
    
    tresh=50
    number = st.number_input('Inserez le numero de client',min_value=0, max_value=999999)
    if number==0:
        st.stop()
    elif number not in df['SK_ID_CURR'].unique():
        st.error('Id client non valide')
        st.stop()

    st.write('Numero de client ', number)

    st.dataframe(df[df['SK_ID_CURR']==number])
    (X,y,y_pred)=result_pred(df,clf)
    index=df.query("SK_ID_CURR == @number").index.values[0]
    col1, col2 = st.columns(2)
    
    col1.metric("TARGET", df.query("SK_ID_CURR == @number")['TARGET'])
    col2.metric("Prediction d'appertenance a la classe 1", round(y_pred[index,1],2))
    
    
    if st.radio(
    "",
    ('Informations général', 'Importance des features dans la prediction'),horizontal=True )=='Informations général':
        client_plot(df,number)
    else:
        (X,y,y_pred)=result_pred(df,clf)
        shap_values_explaination = shap.Explanation(shap_values[1][index,:], feature_names=X.columns.tolist()) 
        fig, ax = plt.subplots()
        ax=shap.plots._waterfall.waterfall_legacy(expected_values[1],shap_values_explaination.values,
                                                  feature_names=X.columns.tolist(),max_display=20)
        st.pyplot(fig)



def button(df):
    page = st.sidebar.radio(
    "selectionnez",
    ('Acceuil','Client', 'Model','Prédiction'),index=0)
    return page
#Renvoie la Target en fonction du client
def result_pred(df,clf):
    
    X=df.drop(['TARGET','SK_ID_CURR'],1).copy()
    y=df['TARGET'].copy()
    results=clf.predict_proba(X)
    return X,y,results

    return plt

#Ecrit les métriques dans la parti modèle
def metrique(X,y,clf,y_pred,y_tresh):


    
    score_recall=recall_score(y_tresh,y).round(2)
    score_précision=precision_score(y_tresh,y).round(2)
    auc_score=roc_auc_score(y_tresh,y).round(2)
    score_f1=fbeta_score(y_tresh,y,beta=2).round(2)
    
    col1, col2, col3,col4 = st.columns(4)
    col1.metric("Recall", score_recall)
    col2.metric("Precision", score_précision)
    col3.metric("AUC score",auc_score)
    col4.metric("F_beta score", score_f1)
#Génere les shap plot de la partie modèle    
def shap_plot_mdl(shap_values,X,df):
    shap.initjs()
    columns=list(df.drop(['SK_ID_CURR','TARGET'],1).columns)
    columns.insert(0,'Informations générales')
    option = st.selectbox(
            'choix de la variable',
            columns)
    shap_value=[shap_values[0],shap_values[1]]
    if option=='Informations générales':
        fig, ax = plt.subplots()
        ax=shap.summary_plot(shap_value, X,max_display=10)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        fig = shap.dependence_plot(option, shap_value[1], X, display_features=X)
        st.pyplot(fig)
    
#Matrice de confusion pour la partie model      
def pred_plot_mdl(y_tresh, y,df):
    #    shap_plot

    conf_mat = metrics.confusion_matrix(y_tresh, y)
    print(conf_mat)
    print()
    col1, col2 = st.columns(2)
    fig, ax = plt.subplots(figsize = (6,4))
    df_cm = pd.DataFrame(conf_mat, index = [label for label in set(y)],
                      columns = [i for i in "01"])
    ax=sns.heatmap(df_cm, annot=True, cmap="Blues")
    ax.set(xlabel='Y_pred', ylabel='Ytrue',title='Matrice de confusion')
    col1.pyplot(fig)
    number = col2.number_input("Entrez l'id client",min_value=0, max_value=999999)
    if number==0:
        st.stop()
    elif number not in df['SK_ID_CURR'].unique():
        st.error('Id client non valide')
        st.stop()
    x=df.query("SK_ID_CURR == @number").index.values[0]
    col2.metric("prediction",y_tresh[x])
    
    

    
#Fonction général de la partie modèle
def model(df,clf,shap_values,expected_values):
    
    st.title('Information général sur le modèle')
    
    (X,y,y_pred)=result_pred(df,clf)
    col1, col2 = st.columns(2)
    col1.write('Roc curve')
    fig_ROC = RocCurveDisplay.from_estimator(clf, X, y)
    
    col1.pyplot(fig_ROC.figure_)
    col2.write('Precision Recall curve')
    fig_ROC = PrecisionRecallDisplay.from_estimator(clf, X, y)
    col2.pyplot(fig_ROC.figure_)
    tresh_min=round((y_pred[:,1].min()*100)+1)
    tresh_max=round((y_pred[:,1].max()*100)-2)
    st.write("Le treshold correspond à la probabilité minal pour que la prédiction d'un cliant pour que celui ci dans la catégorie 1")
    st.write("Ex: un client avoir un prediction de 0.4 pour la classe 1 sera predit:")
    st.write(" - Dans la classe 0 si le treshold est superieur a 20")
    st.write(" - Dans la classe 1 sinon")
    tresh = st.slider(
    'Select a range of values',
    tresh_min, tresh_max)
    st.write('Treshold:', tresh)
    w=lambda x : 0 if x < (tresh/100) else 1
    y_tresh=np.array([w(xi) for xi in y_pred[:,1]])
    st.write("Vous pouvez trouver ci dessous les résultats des différentes métriques ainsi que la matrice de confusion pour le treshold séléctionné")
    st.write("Dans la partie 'Features Importance' vous pourrez trouvez un summary plot montrant les features les plus importantes du model")
    st.write("Vous trouverez pour chaque variable le dependance plot de la variable avec laquelle elle est le plus corrélée")
    metrique(X,y,clf,y_pred,y_tresh)

    if st.radio(
    "",
    ('Predictions', 'Feature importances'),horizontal=True)=='Predictions':
        pred_plot_mdl(y_tresh, y,df)
    else:    
        shap_plot_mdl(shap_values,X,df)
    
    
#Fonction pour la prédiction d'un nouveau clients    
def predict_new(df):
    st.title("Prédiction d'un nouveau client")
    data_name=['application_train.csv','application_test.csv', 'bureau.csv' ,'bureau_balance.csv' , 'credit_card_balance.csv', 'installments_payments.csv', 'POS_CASH_balance.csv', 'previous_application.csv']
    st.write("Pour obtenir la prédiction d'un nouveau client vous devez ajouter 7 Dataframes nommés : ")
    
    uploaded_files = st.file_uploader("Choose a CSV file", type = 'csv', accept_multiple_files=True)
    dicti={}
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        data=pd.read_csv(BytesIO(bytes_data))
        dicti[uploaded_file.name[:-4]]=data.to_json()
    
    if len(dicti) == 8:
            
            response = requests.post('https://p7apirp.herokuapp.com/predict', json=dicti)
            st.write(response.content)
            
            
            response = requests.post('https://p7apirp.herokuapp.com/prepro', json=dicti)
            response=json.loads(response.content.decode("utf-8").replace("'",'"'))
            client_data=pd.DataFrame.from_dict(response,orient='index')

    if len(dicti) == 8:
        try:
            
            response = requests.post('https://p7apirp.herokuapp.com/predict', json=dicti)
            
            response = requests.post('https://p7apirp.herokuapp.com/prepro', json=dicti)
            response=json.loads(response.content.decode("utf-8").replace("'",'"'))
            client_data=pd.DataFrame.from_dict(response,orient='index')
            
            number=client_data['SK_ID_CURR'][0]
            

            
            client_data['TARGET']=[1]
            st.write(client_data)
            df=df[client_data.columns]
            df=pd.concat([df, client_data])

            client_plot(df,number)
            
        except:
            pass
            
    

   
    
    

            
#Page d'acceuil
def acceuil():
    st.title('Ronan PONCET')
    st.title('Projet 7 implementez un model de scoring')
    st.title('Dashboard')
        
    st.write('Bienvenue !')
    st.write("Dans la partie 'Client' vous trouverez toutes les informations des clients deja présent dans la base de donnée et leurs comparaison avec les autres clients")
    st.write("Dans la partie 'Modèle' vous trouverez toutes les informations relative au modèle (Métrique , features importances ...)")
    st.write("Dans la partie 'Prédiction' vous pourrez ajouter les données d'un nouveau clients obtenir sa prédtion pour le modèle ainsi qu'un résumé de ses informations général")
    
    
    
#    score_auc=roc_auc_score(y_pred,y,average='macro')
#    col1, col2 = st.columns(2)
#    col1.metric("TARGET", )
#    col2.metric("Prediction", 2)
#    st.write('goodbye')
    
    
#Fonction principal        
def main():
    
    (df,result,clf,shap_values,expected_values)=read()
    page=button(df)
        
        
    if (page=='Acceuil'):
        acceuil()
    elif (page=='Client'):
        explo_plot(df,clf,shap_values,expected_values)
    elif(page=='Model') :
        model(df,clf,shap_values,expected_values)
    else :
        predict_new(df)
    

if __name__ == '__main__':
    main()
