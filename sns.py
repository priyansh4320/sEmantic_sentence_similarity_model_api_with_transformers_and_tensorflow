import tensorflow as tf
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import cache
from fastapi.encoders import jsonable_encoder

model = tf.keras.models.load_model("precily_sns_mod_complete_regression")

@cache
def st_load():
     return SentenceTransformer("stsb-roberta-large")
st = st_load()

@cache
def embedder(text1 ,text2):
    
    embeddings1 = st.encode(text1,convert_to_tensor=True)
    embeddings2 = st.encode(text2,convert_to_tensor=True)
    embeddings1 = embeddings1.reshape((1,-1))
    embeddings2 = embeddings2.reshape((1,-1))
    data = np.concatenate([embeddings1,embeddings2],axis=1)
    data = data.reshape((1,-1))
    return data 

app = FastAPI()



@app.post("/compare")
async def compare_strings(Sentence1:str, Sentence2:str):
    data = embedder(Sentence1,Sentence2)
    pred = model.predict(data) 
    return {"Similarity Score: ": str(pred[0])}