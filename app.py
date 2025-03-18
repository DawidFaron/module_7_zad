import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import json
import base64
from io import BytesIO
from PIL import Image
from qdrant_client import QdrantClient
from dotenv import dotenv_values
from openai import OpenAI


#
# MAIN

#
# CONSTANCES
MODEL = "welcome_survey_clustering_model_pipeline_v2"
DATA = "welcome_survey_simple_v2.csv"
DATA_JSON = "welcome_survey_name_and_description.json"
EMBEDDING_DIM = 3072
EMBEDDING_MODEL = "text-embedding-3-large"

env = dotenv_values(".env")
### Secrets using Streamlit Cloud Mechanism
# https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']
###

#
# FUNCTIONS
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.session_state.get("OpenAI_key")) 
    
@st.cache_data
def load_data_json():
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        return json.loads(f.read())

@st.cache_data
def get_image_from_base64(cluster):
    data_dict = load_data_json()
    image_encode = base64.b64decode(data_dict[cluster]["image_base64"])
    image_encode_io = BytesIO(image_encode)
    image = Image.open(image_encode_io)
    return image.resize((512, 512))

@st.cache_data 
def get_load_model():
    return load_model(MODEL)
    
@st.cache_data 
def get_predict_model(person_respond):
    model = get_load_model()
    return predict_model(model, data=person_respond)
    
@st.cache_data
def get_predict_model_all_persons():
    model = get_load_model()
    return predict_model(model, data=pd.read_csv(DATA, sep=";"))

def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    return result.data[0].embedding

#
# db
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=env["QDRANT_URL"],
        api_key=env["QDRANT_API_KEY"],
    )

def list_data_from_db(query):
    qdrant_client = get_qdrant_client()
    
    notes = qdrant_client.search(
        collection_name="welcome_survey",
        query_vector=get_embedding(query),
        limit=1,
    )
    result = []
    for note in notes:
        result.append(
            {
                "cluster": note.payload["cluster"],
                "name": note.payload["name"],
                "description": note.payload["description"],
                "image_description": note.payload["image_description"],
                "score": note.score,
            }
        )
    return result

def reset_matched():
    st.session_state["matched"] = ""
    
if not "age" in st.session_state:
    st.session_state["age"] = ""
    
if not "edu_level" in st.session_state:
    st.session_state["edu_level"] = ""
    
if not "fav_animals" in st.session_state:
    st.session_state["fav_animals"] = ""
    
if not "fav_place" in st.session_state:
    st.session_state["fav_place"] = ""
    
if not "gender" in st.session_state:
    st.session_state["gender"] = ""

if not "matched" in st.session_state:
    st.session_state["matched"] = ""


# @st.cache_data
# def get_names_and_descriptions():
#     with open("welcome_survey_name_and_description.json", "r", encoding="utf-8") as f:
#         return json.loads(f.read())

df = get_predict_model_all_persons()
#st.write(df[df["Cluster"] == "Cluster 1"])
#st.dataframe(df)

with st.sidebar:
    st.session_state["age"] = st.selectbox("Wiek", ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", ">=65", "unknown"], on_change=reset_matched)
    st.session_state["edu_level"] = st.selectbox("Wykształcenie", df["edu_level"].unique(), on_change=reset_matched)
    st.session_state["fav_animals"] = st.selectbox("Ulubione zwierzę", df["fav_animals"].unique(), on_change=reset_matched)
    st.session_state["fav_place"] = st.selectbox("Ulubione miejsce", ["Nad wodą", "W lesie", "W górach", "Młodzi Miłośnicy     Zwierząt"], on_change=reset_matched)
    st.session_state["gender"] = st.selectbox("Płeć", ["Kobieta", "Mężczyzna"], on_change=reset_matched)

    person_respond_df = pd.DataFrame([
        {
        "age": st.session_state["age"],
        "edu_level": st.session_state["edu_level"],
        "fav_animals": st.session_state["fav_animals"],
        "fav_place": st.session_state["fav_place"],
        "gender": st.session_state["gender"],
    }
    ])

    
    if not st.session_state.get("OpenAI_key"):
        st.info("Opisz, jak masz ochotę spędzić czas:")
        st.session_state["OpenAI_key"] = st.text_input("Klucz AI: ", type="password")
        st.caption("ℹ️ wpisz swój klucz AI, żebyśmy mogli zatrudnić do tego sztuczną inteligencję!")
        if st.session_state.get("OpenAI_key"):
            st.rerun()
    else:
        st.info("Opisz, jak masz ochotę spędzić czas:")
        query = st.text_input("",placeholder="Opis")
        if st.button("Szukaj"):
            st.session_state["matched"] = list_data_from_db(query.strip())
            #st.write(respond)

st.header("Znajdz znajomych:")

if st.session_state.matched:
    #st.write(st.session_state["matched"])
    st.session_state["age"] = ""
    st.session_state["edu_level"] = ""
    st.session_state["fav_animals"] = ""
    st.session_state["fav_place"] = ""
    st.session_state["gender"] = ""
    
    st.header(f'Najblizej Ci do: {st.session_state["matched"][0]["name"]}')
    st.markdown(f'{st.session_state["matched"][0]["description"]}')
    
    matches_number = len((df[df["Cluster"] == st.session_state["matched"][0]["cluster"]]))
    st.metric("Twoja liczba osób o podobnych zainteresowaniach:", matches_number)
    
    st.image(get_image_from_base64(st.session_state["matched"][0]["cluster"]), use_column_width=True)

    match_level = st.session_state["matched"][0]["score"] * 100
    with st.container():
        st.caption("ℹ️ Wyższy poziom oznacza lepsze dopasowanie.")
        st.metric("Twój poziom dopasowania:", f"{round(match_level, 2)}%")
    
    st.markdown("#### **Jeżeli uważasz, że poniższy opis pasuje do Ciebie lub chciałbyś, albo chciałabyś spróbować             przeżywać takie wartości. To ta    grupa jest dla Ciebie!**")
    data_dict = load_data_json()
    desc = data_dict[st.session_state["matched"][0]["cluster"]]["image_description"]
    st.markdown(f'{data_dict[st.session_state["matched"][0]["cluster"]]["image_description"]}')  

#st.write(person_respond)

if st.session_state["matched"] == "":
    # st.write(st.session_state["age"])
    predict_model_cluster_id = get_predict_model(person_respond_df)["Cluster"][0]
    #st.write(predict_model_cluster_id)
    
    names_and_descriptions = load_data_json()
    st.header(f'Najblizej Ci do: {names_and_descriptions[predict_model_cluster_id]["name"]}')
    st.markdown(f'{names_and_descriptions[predict_model_cluster_id]["description"]}')
    
    matches_number = len((df[df["Cluster"] == predict_model_cluster_id]))
    st.metric("Twoja liczba osób o podobnych zainteresowaniach:", matches_number)
    
    st.image(get_image_from_base64(predict_model_cluster_id), use_column_width=True)
    
    st.markdown("#### **Jeżeli uważasz, że poniższy opis pasuje do Ciebie lub chciałbyś, albo chciałabyś spróbować             przeżywać takie wartości. To ta    grupa jest dla Ciebie!**")
    
    data_dict = load_data_json()
    desc = data_dict[predict_model_cluster_id]["image_description"]
    st.markdown(f'{data_dict[predict_model_cluster_id]["image_description"]}')  
    
      


          










