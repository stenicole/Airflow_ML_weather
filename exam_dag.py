from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.sensors.filesystem import FileSensor

import os
import requests
import json
import datetime
import pandas as pd
import time

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump



my_dag = DAG(
    dag_id='exam_dag',
    description='My first openweather DAG',
    tags=['tutorial', 'datascientest'],
    schedule_interval='* * * * *',
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0),    
    },
    catchup=False
)



def recup_data():
    # création du dossier  'app/raw_files' de destination des fichiers résultats des requêtes
    filepath = '/app/raw_files'
    if os.path.exists(filepath) == False:
        os.makedirs(filepath, mode = 511, exist_ok= False)
    # positionnement dans le dossier '/app/raw_files'
    os.chdir(filepath)
    # création de la liste des villes pour lesquelles les données météo vont être demandées
    city_db = ['paris', 'london', 'washington']
    
    # pour chaque ville de la liste précédente
    for city in city_db:
        # requête des données météo
        r = requests.get('https://api.openweathermap.org/data/2.5/weather',
        params= {
        'q': city,
        'appid': '90be6b9a92b011fcbe9f458fce8b9632'
            }
        )
        result = r.text
        output = json.loads(result)    
        print(output)
       
        # Création de la Variable cities
        cities = Variable.set(key=city, value=output)
        # pause de 2s pour que chaque fichier .json ait un nom différent
        time.sleep(2)

        # création du nom du fichier dont le nom correspond à la date et l'heure de la récolte
        filename = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')+'.json'
        # Remplissage du fichier avec les données météos récoltées
        with open(filename, 'a') as file:
            json.dump(output,file,indent=2)
    


def transform_data_into_csv():
    filename = 'data.csv'
    parent_folder = '/app/raw_files'
    # positionnement dans le dossier '/app/raw_files'
    os.chdir(parent_folder)

    files = sorted(os.listdir(parent_folder), reverse=True)
    files = files[:20]

    dfs = []

    for f in files:
        with open(os.path.join(parent_folder, f), 'r') as file:
            data_temp = json.load(file)        
            dfs.append(
                {
                    'temperature': data_temp['main']['temp'],
                    'city': data_temp['name'],
                    'pression': data_temp['main']['pressure'],
                    'date': f.split('.')[0]
                }
            )

    df = pd.DataFrame(dfs)
        
    # création du dossier  'app/clean_data' de destination des fichiers '.csv' s'il n'existe pas
    filepath = '/app/clean_data'
    if os.path.exists(filepath) == False:
        os.makedirs(filepath, mode = 511, exist_ok= False)

    df.to_csv(os.path.join('/app/clean_data', filename), index=False)

def transform_data_into_csv_full():
    filename = 'fulldata.csv'
    parent_folder = '/app/raw_files'
    files = sorted(os.listdir(parent_folder), reverse=True)
    Variable.set(key="nb_f", value=len(files))
    nb_f = Variable.get(key="nb_f")
    files = files[:60]
    
    dfs = []

    for f in files:
        with open(os.path.join(parent_folder, f), 'r') as file:
            data_temp = json.load(file)        
        dfs.append(
                {
                    'temperature': data_temp['main']['temp'],
                    'city': data_temp['name'],
                    'pression': data_temp['main']['pressure'],
                    'date': f.split('.')[0]
                }
            )

    df = pd.DataFrame(dfs)

    

    df.to_csv(os.path.join('/app/clean_data', filename), index=False)


def compute_model_score(model, X, y):
    # computing cross val
    cross_validation = cross_val_score(
        model,
        X,
        y,
        cv=3,
        scoring='neg_mean_squared_error')

    model_score = cross_validation.mean()

    return model_score



def prepare_data(path_to_data='/app/clean_data/fulldata.csv'):
    # reading data
    df = pd.read_csv(path_to_data)
    # ordering data according to city and date
    df = df.sort_values(['city', 'date'], ascending=True)

    dfs = []

    for c in df['city'].unique():
        df_temp = df[df['city'] == c]

        # creating target
        df_temp.loc[:, 'target'] = df_temp['temperature'].shift(1)

        # creating features
        for i in range(1, 10):
            df_temp.loc[:, 'temp_m-{}'.format(i)
                        ] = df_temp['temperature'].shift(-i)

        # deleting null values
        df_temp = df_temp.dropna()

        dfs.append(df_temp)

    # concatenating datasets
    df_final = pd.concat(
        dfs,
        axis=0,
        ignore_index=False
    )

    # deleting date variable
    df_final = df_final.drop(['date'], axis=1)

    # creating dummies for city variable
    df_final = pd.get_dummies(df_final)

    X = df_final.drop(['target'], axis=1)
    y = df_final['target']
    
    X.to_pickle('/app/clean_data/features.pkl')
    y.to_pickle('/app/clean_data/target.pkl')

    X = pd.read_pickle('/app/clean_data/features.pkl')
    y = pd.read_pickle('/app/clean_data/target.pkl')
    
    score_lr = compute_model_score(LinearRegression(), X, y)
    score_dt = compute_model_score(DecisionTreeRegressor(), X, y)
    score_rf = compute_model_score(RandomForestRegressor(), X, y)
    
    Variable.set(key="score_lr", value=score_lr)
    Variable.set(key="score_dt", value=score_dt)
    Variable.set(key="score_rf", value=score_rf)
    

def train_and_save_model(model, X, y, path_to_model='/app/model.pckl'):
    # training the model
    model.fit(X, y)
    # saving model
    print(str(model), 'saved at ', path_to_model)
    dump(model, path_to_model)

def selection_model():
   X = pd.read_pickle('/app/clean_data/features.pkl')
   y = pd.read_pickle('/app/clean_data/target.pkl')

   score_lr = Variable.get(key="score_lr")
   score_dt = Variable.get(key="score_dt")
   score_rf = Variable.get(key="score_rf")
   
   score = [score_lr, score_dt, score_rf]   

   if score_lr == max(score):
        train_and_save_model(
            LinearRegression(),
            X,
            y,
            '/app/clean_data/best_model.pickle'
        )
   elif score_dt == max(score):
        train_and_save_model(
            DecisionTreeRegressor(),
            X,
            y,
            '/app/clean_data/best_model.pickle'
        )
   else:
        train_and_save_model(
            RandomForestRegressor()(),
            X,
            y,
            '/app/clean_data/best_model.pickle'
        )


#task_1 = PythonOperator(
#     task_id='recup_meteo_data',
#     python_callable=recup_data,
#     dag=my_dag
#)

# Vérification présense répertoire des fichier '.json'

sensor_json = FileSensor(
    task_id="check_json_folder",
    fs_conn_id="json_connection",
    filepath="raw_files",
    poke_interval=30,
    dag=my_dag,
    timeout=5 * 30,
    mode='reschedule'
)

task_2 = PythonOperator(
    task_id='transform_data_csv',
    python_callable=transform_data_into_csv,
    dag=my_dag
)

sensor_csv = FileSensor(
    task_id="check_csv_file",
    fs_conn_id="csv_connection",
    filepath="data.csv",
    poke_interval=30,
    dag=my_dag,
    timeout=5 * 30,
    mode='reschedule'
)


task_3 = PythonOperator(
    task_id='transform_data_full_csv',
    python_callable=transform_data_into_csv_full,
    dag=my_dag
)

sensor_fullcsv = FileSensor(
    task_id="check_fullcsv_file",
    fs_conn_id="csv_connection",
    filepath="fulldata.csv",
    poke_interval=30,
    dag=my_dag,
    timeout=5 * 30,
    mode='reschedule'
)

task_4 = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    dag=my_dag
)

task_5 = PythonOperator(
    task_id='selection_model',
    python_callable=selection_model,
    dag=my_dag
)
#task_1 >> sensor_json  
sensor_json >>  [task_3, task_2]
sensor_csv << task_2
sensor_fullcsv << task_3
sensor_fullcsv  >> task_4
task_4 >> task_5

