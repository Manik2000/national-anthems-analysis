import os
import pickle

import numpy as np
import pandas as pd
from openai import OpenAI


client = OpenAI()


def get_embedding(text, model="text-embedding-3-large"):
   return client.embeddings.create(input=[text], model=model).data[0].embedding


if __name__ == '__main__':
   
   path = os.path.join("data", "for_embedding.csv")
   df = pd.read_csv(path) 
   lyrics = df.lyrics  
   string = " ".join(list(lyrics))

   openai_embedding = np.array(list(map(get_embedding, lyrics))) 
   embedding_dict = dict()
   embedding_dict["embedding"] = openai_embedding
   embedding_dict["lyrics"] = lyrics
   embedding_dict["iso_2"] = df.iso_2
   embedding_dict["iso_a3"] = df.iso_a3
   
   with open(os.path.join("data", "openai_embedding.pkl"), "wb") as f:
      pickle.dump(embedding_dict, f)
