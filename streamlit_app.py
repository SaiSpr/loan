#-----------------------#
# IMPORT DES LIBRAIRIES #
#-----------------------#

import streamlit as st
import joblib
import streamlit.components.v1 as components
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
# import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)
import shap
import requests as re
import numpy as np
import pickle
import json
import pandas as pd
from streamlit_echarts import st_echarts
import seaborn as sns 
plt.style.use('ggplot')


st.title("Bank Loan Detection Web App")

st.image("image.jpg")



st.sidebar.header('Select the Client_id:')

sender_name = 1
receiver_name = 2

types = 4


amount = 4
oldbalanceorg = 5
newbalanceorg= 6
oldbalancedest=7
newbalancedest= 8
isflaggedfraud = 0




  
st.title("****Calculating the probability that a customer will repay their credit or not****")  

# logo sidebar 
st.sidebar.image("home_credit.png", use_column_width=True)
  
 
# Read 
list_file = open('cols_shap_local.pickle','rb')
cols_shap_local = pickle.load(list_file)
print(cols_shap_local)



#df_test_prod = pd.read_csv('df_test_ok_prod_100.csv', index_col=[0])
df_test_prod = pd.read_csv('df_test_ok_prod_100_V7.csv', index_col=[0])
df_test_prod['LOAN_DURATION'] = 1/df_test_prod['PAYMENT_RATE']
df_test_prod.drop(columns=['TARGET'], inplace=True)
df_test_prod_request  = df_test_prod.set_index('SK_ID_CURR')



df_train = pd.read_csv('df_train_prod_1.csv', index_col=[0])
df_train['LOAN_DURATION'] = 1/df_train['PAYMENT_RATE']

# Liste clients id sidebar 
list_client_prod = df_test_prod['SK_ID_CURR'].tolist()
client_id = st.sidebar.selectbox("Client Id list",list_client_prod)
client_id = int(client_id)

st.header(f'Credit request result for client {client_id}')

    
step = client_id
    

    
#################################################    
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)    
    
#################################################
def explain_plot(id, pred):
    
    pipe_prod = joblib.load('LGBM_pipe_version7.pkl')
    df_test_prod_1 = df_test_prod.reset_index(drop=True)
    df_test_prod_request_1 = df_test_prod_1.reset_index().set_index(['SK_ID_CURR', 'index'])
    df_shap_local = df_test_prod_request_1[df_test_prod_request_1.columns[df_test_prod_request_1.columns.isin(cols_shap_local)]]
    values_id_client = df_shap_local.loc[[id]]
    

    explainer = shap.TreeExplainer(pipe_prod.named_steps['LGBM'])
      
    observation = pipe_prod.named_steps["transform"].transform(df_shap_local)
    observation_scale = pipe_prod.named_steps["scaler"].transform(observation)

    shap_values = explainer.shap_values(observation_scale)

    if pred == 1:
        p = st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][values_id_client.index[0][1],:],values_id_client))
    else:
        p = st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0][values_id_client.index[0][1],:],values_id_client))
    return p
###################################################  
  
  
# Filtrer les clients rembourser et non rembourser 
df_train_rembourse = df_train[df_train['TARGET']== 0.0]
df_train_not_rembourse = df_train[df_train['TARGET']== 1.0]

# Sélectionner les colonnes pour le dashboard
cols_dashbord = ['SK_ID_CURR','AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AGE', 'LOAN_DURATION']



df_train_not_rembourse = df_train_not_rembourse[cols_dashbord]
df_train_rembourse = df_train_rembourse[cols_dashbord]

###################################################  
  
  
  
if st.button("Detection Result"):
    values = {
    "step": step,
    }


    st.write(f"""### These are the details:\n

    Client Id is: {step}\n
    
    type of client_id is : {type(step)}
    
    Values is : {values}
    
    Values tyep is : {type(values)}

                """)

    res = re.post( url ="http://127.0.0.1:8000/predict", data = json.dumps(values))

    json_str = json.dumps(res.json())
    
        
    st.write(json_str)
    st.write(type(json_str))
    resp = json.loads(json_str)
    
#     prediction = res

    st.write(res)
    st.write(type(res))
    
    st.write(resp)
    st.write(type(resp))
    
    pred = resp["prediction"]

    probability_value_0 = round(resp["probability_0"] * 100,2)
    probability_value_1 = round(resp["probability_1"] * 100,2)


    st.header(f'*Result of the credit application for the customer {client_id} is:*')

    st.write(pred)
    st.write(type(pred))
    if pred == 1:
      st.error('Loan Refused')
      option_1 = {
            "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
            "series": [
                {
                    "name": "Pressure",
                    "type": "gauge",
                    "axisLine": {
                        "lineStyle": {
                            "width": 10,
                        },
                    },
                    "progress": {"show": "true", "width": 10},
                    "detail": {"valueAnimation": "true", "formatter": "{value}"},
                    "data": [{"value": probability_value_1, "name": "Probabilité %"}],
                }
            ],
        }

      st_echarts(options=option_1, width="100%", key=0)
      st.header(f'*The data that most influenced the calculation of the prediction for the client {client_id} is:*')

      explain_plot(client_id, pred)
    else:
        st.success('Loan Approved')
        option = {
            "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
            "series": [
                {
                    "name": "Pressure",
                    "type": "gauge",
                    "axisLine": {
                        "lineStyle": {
                            "width": 10,
                        },
                    },
                    "progress": {"show": "true", "width": 10},
                    "detail": {"valueAnimation": "true", "formatter": "{value}"},
                    "data": [{"value": probability_value_0, "name": "Probabilité %"}],
                }
            ],
        }

        st_echarts(options=option, width="100%", key=0)

        st.header(f'*The data that most influenced the calculation of the prediction for the client {client_id} is:*')
        explain_plot(client_id, pred)



    st.header("*The most significant variables in descending order and which have a high predictive power.*")

    st.image("Shap_features_global.png", use_column_width=True)

    df_informations_client = df_test_prod[['SK_ID_CURR','CODE_GENDER','AGE', 'FLAG_OWN_CAR','FLAG_OWN_REALTY', 'CNT_CHILDREN',
    'AGE','AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3','LOAN_DURATION']]




    st.header(f"*Customer's personal information {client_id} :*")
    df_test_visu = df_test_prod[['SK_ID_CURR','AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AGE', 'LOAN_DURATION']]

    st.write(df_informations_client[df_informations_client['SK_ID_CURR']==client_id].transpose())


    #plot 
    st.header(f'*Descriptive information about the customer {client_id}*'
    )
    list_cols_dashboard = ['AMT_GOODS_PRICE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AGE', 'LOAN_DURATION',]
    for col in list_cols_dashboard:


        #fig = ff.create_distplot(col,df_train, bin_size=[.1, .25, .5])
        #st.plotly_chart(fig, use_container_width=True)
        #st.pyplot()



         plt.figure(figsize=(12,8))
         plt.gca().set_title(f'{col} At', size=30)
         sns.distplot(df_train_rembourse[[col]], label='Rembourse', color='green', hist=False, bins=8)
         sns.distplot(df_train_not_rembourse[[col]], label='Non_rembourse', color='red', hist=False, bins=8)
         plt.axvline(x=float(df_test_visu[df_test_visu['SK_ID_CURR']== client_id][[col]].values),
                     color='blue', ls=':', lw=4, label=client_id)
      #plt.annotate('Le client', xy = (000000.175, 100001), xytext = (1.75e-6, 100001),
                     # arrowprops = {'facecolor': 'red', 'shrink': 0.1})
         plt.legend()
      # Plot!
      #st.plotly_chart(fig, use_container_width=True)
         st.pyplot()
            
    option_3 = {
            #"title": {"text": "Comparaison du client avec la base de données"},
            "legend": {"data": ["（Payment default）", "（Non default payment）", f"the customer {client_id}"]},
            "radar": {
                "indicator": [
                    {"name": "（Age）", "max": 70},
                    {"name": "（AMT_GOODS_PRICE）", "max": 1800000.00},
                    {"name": "（AMT_CREDIT）", "max": 2102490.00},
                    {"name": "（AMT_INCOME_TOTAL）", "max": 540000.00},
                    {"name": "（AMT_ANNUITY）", "max": 74416.50},
                    {"name": "（LOAN_DURATION）", "max": 32.21},
                ]
            },
            "series": [
                {
                    "name": "（Client_id vs. Database）",
                    "type": "radar",
                    "data": [
                        {
                            "value": [43.71 ,488972.41, 557778.53, 165611.76,  26481.74,20.76 ],
                            "name": "（Payment default）",
                        },
                        {
                            "value": [40.28 ,542738.51, 602651.16, 169077.47, 27163.73, 21.68 ],
                            "name": "（Non default payment）",
                        },
                        {
                            "value": list(df_test_prod[df_test_prod['SK_ID_CURR']==client_id][["AGE","AMT_GOODS_PRICE","AMT_CREDIT","AMT_INCOME_TOTAL","AMT_ANNUITY","LOAN_DURATION"]].squeeze())
    ,
                            "name": f"le client {client_id}",
                        },
                    ],
                }
            ],
        }
    st_echarts(option_3, height="600px")
