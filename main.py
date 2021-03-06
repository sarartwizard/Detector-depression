from json import encoder
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import model_from_json

primaryColor="#2214c7"
backgroundColor="#ffffff"
secondaryBackgroundColor="#e8eef9"
textColor="#000000"
font="sans serif"
st.title('Depression Detector :pill:')

st.write('''
### Cette application diagnostique la presence de trouble mental chez une personne de plus de 15ans :computer:
''')

st.sidebar.header("les parametres d'entrée")


genre = st.sidebar.slider('Homme = 0 / Femme = 1', 0, 1, 1)
Age = st.sidebar.slider('Votre age', 15, 60, 25)
Trouble_du_Sommeil = st.sidebar.slider('Trouble du sommeil', 0, 10, 5)
Fatigue_intense = st.sidebar.slider('Fatigue intense',0,10,8)
Ralentissement_psychomoteur_général= st.sidebar.slider('Ralentissement psychomoteur général',0,10,7)
Perte_de_confianceen_soi = st.sidebar.slider('perte de confiance en soi', 0, 10, 5)
Anxiété= st.sidebar.slider('Anxiété',0,10,5)
Irritabilite_frustration = st.sidebar.slider('Irritabilité et frustration', 0, 10, 4)
Troubles_de_la_mémoire = st.sidebar.slider('trouble de la memoire',0,10,8)
Douleur_physique_sans_causes= st.sidebar.slider('Douleur physique sans causes',0,10,5)
envies_suicidaires= st.sidebar.slider('envies suicidaires',0,10,5)
modififcation_de_lappetit = st.sidebar.slider('modification de lappetit', 0, 10, 5)
Fausses_croyances= st.sidebar.slider('Fausses croyances',0,10,5)
Hallucination= st.sidebar.slider('Hallucination',0,10,5)
interval_de_temps = st.sidebar.slider('interval de temps de trouble depressif', 0, 2, 1)
variablededepre = st.sidebar.slider('tristesse', 0, 10,5)
Hyperactivité = st.sidebar.slider('Hyperactivité', 0, 10, 0)
bonheur_intense = st.sidebar.slider('bonheur intense', 0, 10, 5)
estime_de_soi_démesuré = st.sidebar.slider('estime de soi démesuré', 0, 10, 3)
accéleration_de_la_pensé = st.sidebar.slider('accéleration de la pensé', 0, 10, 4)
grande_distraction = st.sidebar.slider('grande distraction', 0, 10, 5)
comportement_a_risque = st.sidebar.slider('comportement a risque', 0, 10, 8)
energie_debordante = st.sidebar.slider('energie debordante', 0, 10, 5)
dimunition_du_besoin_de_dormir = st.sidebar.slider('dimunition du besoin de dormir', 0, 10, 4)
variableB = st.sidebar.slider('durée de la manie', 0, 2, 1)
interval_de_temps2 = st.sidebar.slider('interval de temps des symptomes euphoriques ', 0, 2, 0)


def user_input():




    data = {
            'genre': genre,
            'Age':Age,
            'Trouble_du_Sommeil': Trouble_du_Sommeil,
            'Fatigue_intense': Fatigue_intense,
            'Ralentissement_psychomoteur_général': Ralentissement_psychomoteur_général,
            'Perte_de_confianceen_soi': Perte_de_confianceen_soi,
            'envies_suicidaires':envies_suicidaires,
            'Anxiété':Anxiété,
            'Irritabilite_frustration': Irritabilite_frustration,
            'Troubles_de_la_mémoire': Troubles_de_la_mémoire,
            'Douleur_physique_sans_causes': Douleur_physique_sans_causes,
            'modififcation_de_lappetit': modififcation_de_lappetit,
            'Fausses_croyances' : Fausses_croyances,
            'Hallucination' : Hallucination,
            'interval_de_temps' : interval_de_temps,
            'variablededepre' : variablededepre,
            'Hyperactivité' : Hyperactivité,
            'bonheur_intense' : bonheur_intense,
            'estime_de_soi_démesuré' : estime_de_soi_démesuré,
            'accéleration_de_la_pensé' : accéleration_de_la_pensé,
            'grande_distraction' : grande_distraction,
            'comportement_a_risque' : comportement_a_risque,
            'energie_debordante' : energie_debordante,
            'dimunition_du_besoin_de_dormir' : dimunition_du_besoin_de_dormir,
            'variableB' : variableB,
            'interval_de_temps2': interval_de_temps2

    }


    parametres_depression =pd.DataFrame(data,index=[0])
    return parametres_depression

df=user_input()
nmp=df.to_numpy()
print(df)

st.write('''Veuillez évaluer vos symptomes sur une échelle de 1 à 10. Plus le chiffre est élevé, plus le symptome est intense.''')
st.write(''' 0 :arrow_right:  Jamais''')
st.write(''' 1 et 3 :arrow_right:  Rarement''')
st.write('''entre 3 et 5 :arrow_right: Souvent''')
st.write('''entre 5 et 8 :arrow_right: Tres souvent''')
st.write('''entre 8 et 10 :arrow_right: Tout le temps ''')


import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import csv


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
data = pd.read_excel("DatasetPA.xlsx")
data.drop(columns='ID', inplace = True)
data = data.drop(data.index[0])
data = data.sample(frac=1).reset_index(drop=True)
data["A"] = data["A"].replace(",", ".")
data["B"] = data["B"].replace(",", ".")
encoder = OrdinalEncoder()
labels = encoder.fit_transform((np.array(data["Diagnostique"]).reshape(-1,1)))
features = np.array(data.drop(columns="Diagnostique"))
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.3, random_state = 42 ,stratify = labels )
NUM_CLASSES = 6


def Build_Model():
    model = keras.Sequential([
        keras.layers.Dense(26),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(80, activation='relu'),
        keras.layers.Dense(60, activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    opt= tf.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
    return model

model = Build_Model()



history = model.fit(X_train, y_train , validation_data=(X_test,y_test),  epochs=100, batch_size= 1000)

# 
# st.write(df)
# 
# # charger le modele pour faire des prédictions sur des nouvelles données
# json_file = open("model_MLPCLASSIFER.json")
# #json_file = open("C:/Users/nadou/PycharmProjects/Projet annuel/Detector-depression/model_MLPCLASSIFER.json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# 
# import h5py    
#     
# rf = h5py.File('model_MLPCLASSIFER.h5','r+')   
# 
# model.load_weights(rf)
# #model.load_weights('C:/Usersvnadou/PycharmProjects/Projet annuel/Detector-depression/model_MLPCLASSIFER.h5')
# 
# 
# print(" -------  The model is  loaded from disk  -------")
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 
# 
# # Tester sur de nouvelles données (données de test)
# from sklearn.datasets import make_blobs
# 
# xnew, _ = make_blobs(n_samples=1, centers=2, n_features=26, random_state=1)
# ynew = model.predict_proba(xnew)
# 
# 
# for i in range(len(xnew)):
# 	print("X=%s, Predicted=%s" % (xnew[i], ynew[i]))
# 
# 
# ynew = np.argmax(ynew, axis= 1)
# 
# for i in range(len(xnew)):
# 	print("X=%s, Predicted=%s" % (xnew[i], ynew[i]))


# convertir le label en maladie 

liste = ["Vous avez peut etre une Dépression bipolaire de type un, nous vous conseillons de voir un spécialiste de la santé :fearful:", 
	 "Vous avez peut etre une Dépression bipolaire de type deux, nous vous conseillons de voir un spécialiste de la santé :fearful: ",
	 "Bonne nouvelle vous n'avez pas de probleme de santé mental :tada: ... pour le moment",
	 "vous souffrez peut etre de dépression récurrente brève, nous vous conseillons de voir un spécialiste de la santé  :fist:",
	 "vous souffrez peut etre de dysthymie, nous vous conseillons de voir un spécialiste de la santé :+1:",
	 "vous souffrez peut etre de troube depressif psychotique, nous vous conseillons de consulter en urgence un spécialiste de la santé :hospital:"
	]


Xnew = nmp
ynew = model.predict(Xnew)
ynew = np.argmax(ynew, axis= 1)

ynew = ynew.item()
st.subheader(liste[ynew])


import boto3

def putNewRes(genre, age, sommeil, fatigue, ralentissement, confiance, anxiete, irritabilite, memoire,
              douleur, suicide, appetit, croyance, hallucination, temps, vardepre, hyperactif, bonheur, estime,acceleration,
              distraction, risque, energie, dormir, varb, it2, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-2', aws_access_key_id='AKIAU7JVIRZH7TLT5VZL', aws_secret_access_key='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')

    newid = getNewId()
    newid2 = str(newid)
    table = dynamodb.Table('ppd_results')
    response = table.put_item(
       Item={
            'id': newid2,
            'genre': genre,
            'age': age,
            'sommeil': sommeil,
            'fatigue': fatigue,
            'ralentissement': ralentissement,
            'confiance': confiance,
            'anxiete': anxiete,
            'irritabilite': irritabilite,
            'memoire': memoire,
            'douleur': douleur,
            'suicide': suicide,
            'appetit': appetit,
            'croyance': croyance,
            'hallucination': hallucination,
            'temps': temps,
            'vardepre': vardepre,
            'hyperactif': hyperactif,
            'estime': estime,
            'acceleration': acceleration,
            'distraction': distraction,
            'risque': risque,
            'energie': energie,
            'dormir': dormir,
            'varb': varb,
            'it2': it2
        }
    )
    return response

def getNewId(dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-2', aws_access_key_id='AKIAU7JVIRZH7TLT5VZL',
                                  aws_secret_access_key='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')
    table = dynamodb.Table('ppd_results')
    response = table.scan()
    return response['Count']

putNewRes(genre,Age,Trouble_du_Sommeil,Fatigue_intense,Ralentissement_psychomoteur_général,Perte_de_confianceen_soi,    
	Anxiété,Irritabilite_frustration,Troubles_de_la_mémoire,Troubles_de_la_mémoire,Douleur_physique_sans_causes,envies_suicidaires
	,modififcation_de_lappetit,Fausses_croyances,Hallucination,interval_de_temps,variablededepre,Hyperactivité,bonheur_intense, estime_de_soi_démesuré,
	accéleration_de_la_pensé,grande_distraction,comportement_a_risque,energie_debordante,dimunition_du_besoin_de_dormir,variableB,interval_de_temps2)


# depression = pd.read_excel('C:/Users/nadou/OneDrive/Documents/Depression.xlsx')
# rfc = RandomForestClassifier(n_estimators=100)
#
#
# train, test = train_test_split(depression, test_size=0.2)
# target_name = train["Diagnostique"]
# train_feat = train.iloc[:,:10]
# train_targ = train["Diagnostique"]
#
# test_feat = test.iloc[:,:10]
# test_targ = test["Diagnostique"]
#
# rfc.fit(train_feat, train_targ)
#
# prediciton = rfc.predict(df)
#
# st.subheader("votres état mental:")
# st.write(prediciton)
