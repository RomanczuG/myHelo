import requests
import json
import streamlit as st
import pandas as pd
import openai
import numpy as np
import faiss
import re

from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv
import os

stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"
])


load_dotenv()

openai_api_key = os.getenv("openai")
openai.api_key = openai_api_key

def basic_lemmatizer(word):
    endings = ['ing', 'ed', 'es', 's']
    for ending in endings:
        if word.endswith(ending):
            return word[:-len(ending)]
    return word

def preprocess_url(url):

    url = re.sub(r'https?://', '', url)
    tokens = re.split(r'[:/?&=]', url)
    tokens = [token for token in tokens if not token.startswith(':')]
   
    return ' '.join(tokens)

def preprocess_description(description):
    if description is None:
        return ""
    
    description = description.lower()
    description = re.sub(r'[^\w\s]', '', description)
    description = " ".join([basic_lemmatizer(word) for word in description.split() if word not in stop_words])
    
    return description

def preprocess(text, text_type='description'):
    if text_type == 'url':
        return preprocess_url(text)
    else:
        return preprocess_description(text)
    
# import json file
with open('myHelo_API.json') as f:
    DATA = json.load(f)

def get_embeddings(texts):
    # Preprocess the texts
    # texts = [preprocess(text) for text in texts]
    # Get the embeddings from OpenAI
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

def generate_response(api_details):
    summary = "I found the following API details:\n"
    for detail in api_details:
        summary += f"Method: {detail['Method']}, URL: {detail['URL']}, Parameters: {detail['Parameters']}, Description: {detail['Description']}\n"
    
    # Generate a more coherent and user-friendly message with OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [
            {"role": "system", "content": "You are a helpful assistant that provides really short information about myhELO API."},
            {"role": "user", "content": summary}
        ]
    )
    
    return response.choices[0].message.content

def getResponse(question):
    # Define the number of similar events to retrieve
    k = 4

    # Load the data
    data = pd.DataFrame(DATA)
    data = data[["method", "url", "parameters", "example_response", "description"]]
    data = data.reset_index(drop=True)

    urls = [preprocess(url, text_type='url') for url in data["url"].tolist()]
    descriptions = [preprocess(desc, text_type='description') for desc in data["description"].tolist()]
    
    # Get the embeddings for the descriptions and urls
    description_embeddings = get_embeddings(urls)
    url_embeddings = get_embeddings(descriptions)
    
    # Concatenate the description and url embeddings
    weight_for_url = 3  # for example
    weighted_url_embeddings = weight_for_url * url_embeddings
    concatenated_embeddings = np.concatenate((weighted_url_embeddings, description_embeddings), axis=1)

    # concatenated_embeddings = np.concatenate((description_embeddings, url_embeddings), axis=1)
    
    # Index using FAISS
    index = faiss.IndexFlatL2(concatenated_embeddings.shape[1])
    index.add(concatenated_embeddings)
    
    # Preprocess the question and get its embedding
    # query_embedding = get_embeddings([question])
    # query_embedding = np.repeat(query_embedding, 2, axis=1)
    query_embedding = get_embeddings([question])
    weighted_query_embedding = np.concatenate((weight_for_url * query_embedding, query_embedding), axis=1)

    # Search for the most similar events to the query
    D, I = index.search(weighted_query_embedding, k)
    api_details = []
    for idx in I[0][:4]:
        row = data.iloc[idx]
        method = row['method']
        url = row['url']
        parameters = row['parameters'] if row['parameters'] else "None"
        if row['example_response']:
            example_response = row['example_response']
        else:
            example_response = "None"
        # example_response = row['example_response'][:50] + "..." if len(row['example_response']) > 50 else row['example_response']
        description = row['description']

        response = generate_response(api_details)

        api_details.append({
            "Method": method,
            "URL": url,
            "Parameters": parameters,
            "Example Response": example_response,
            "Description": description
        })

    return question, api_details, response

    
    # Formatting the output
    # content = f"**Question:** {question}\n\n---\n\n### Here are the top most similar API calls to your question:\n"
    
    # for idx in I[0][:4]:
    #     row = data.iloc[idx]
    #     method = row['method']
    #     url = row['url']
    #     parameters = row['parameters'] if row['parameters'] else "None"
    #     example_response = row['example_response'][:50] + "..." if row['example_response'] else "None"  # Displaying only the first 50 characters for brevity
    #     content += f"- **Method:** {method}\n  **URL:** {url}\n  **Parameters:** {parameters}\n  **Example Response:** {example_response}\n\n"
    
    # return content


        



with st.sidebar:

    st.markdown("""
# MVP showcasing implementation of Chatbot for myhELO API
### Not made for public use    
#### Here you can see API Docs            
                """)
        
    st.json(DATA)


st.markdown("""
## Ask a question about api call
                """)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("user"):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    question, api_details, response = getResponse(prompt)

    with st.chat_message("assistant"):
        # st.subheader(f"Question: {question}")
        # st.write("---")
        st.markdown(response)
        st.subheader("Top similar API calls to your question:")
        for detail in api_details:
            st.text(f"Method: {detail['Method']}")
            st.text(f"URL: {detail['URL']}")
            st.text(f"Parameters: {detail['Parameters']}")
            st.text(f"Description: {detail['Description']}")
            
            with st.expander("Click to view example response"):
                st.text(detail['Example Response'])
                
            st.write("---")

    st.session_state.messages.append({"role": "assistant", "content": "Displayed top similar API calls."})