import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API key
client = OpenAI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load the CSV file
csv_file = 'deal_sourcing_20240828_132806.csv'
df = pd.read_csv(csv_file)

# Function to get embeddings from OpenAI
def get_embeddings(text, model="text-embedding-3-small"):
    text = str(text).replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Load existing embeddings if the embeddings file exists
embeddings_file = 'embeddings.csv'
if os.path.exists(embeddings_file):
    df_embeddings = pd.read_csv(embeddings_file)
    df_embeddings['embedding'] = df_embeddings['embedding'].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
    df = df.merge(df_embeddings[['Name', 'embedding']], on='Name', how='left')
else:
    # Get embeddings from the "Description" column and save embeddings file
    df['embedding'] = df['Description'].apply(get_embeddings)
    df_embeddings = df.copy()
    df.to_csv(embeddings_file, index=False)

# Load the DataFrame with embeddings
df_embeddings = pd.read_csv(embeddings_file)
df_embeddings['embedding'] = df_embeddings['embedding'].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
embeddings_matrix = np.vstack(df_embeddings['embedding'].values)

# Function to search companies based on description
def search_companies(query, df):
    query_embedding = get_embeddings(query)
    similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]
    df['similarity'] = similarities
    return df.sort_values(by='similarity', ascending=False)

# Streamlit app
query = st.text_input('Enter search query:')
if query:
    results = search_companies(query, df_embeddings)
    st.write(results[['Name', 'Description', 'Action','similarity']])

