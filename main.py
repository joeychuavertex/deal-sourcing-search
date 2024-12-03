import streamlit as st
import pandas as pd
import openai
import os
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API key
openai.openai_api_key = 'your_openai_api_key'

# Load the CSV file
csv_file = 'deal_sourcing_20240828_132806.csv'
df = pd.read_csv(csv_file)

# Embeddings file
embeddings_file = 'embeddings.csv'

# Load existing embeddings if the file exists
if os.path.exists(embeddings_file):
    embeddings_df = pd.read_csv(embeddings_file)
else:
    embeddings_df = pd.DataFrame(columns=['text', 'embedding'])

# Function to get embeddings from OpenAI or from the CSV file
def get_embedding(text):
    global embeddings_df
    if text in embeddings_df['text'].values:
        return embeddings_df.loc[embeddings_df['text'] == text, 'embedding'].values[0]
    else:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        embedding = response['data'][0]['embedding']
        new_row = pd.DataFrame({'text': [text], 'embedding': [embedding]})
        embeddings_df = pd.concat([embeddings_df, new_row], ignore_index=True)
        embeddings_df.to_csv(embeddings_file, index=False)
        return embedding

# Function to search companies based on description
def search_companies(query, df):
    query_embedding = get_embedding(query)
    df['embedding'] = df['description'].apply(lambda x: get_embedding(x))
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
    return df.sort_values(by='similarity', ascending=False)

# Streamlit app
st.title('Company Description Search')

query = st.text_input('Enter search query:')
if query:
    results = search_companies(query, df)
    st.write(results[['company_name', 'description', 'similarity']])