import os
import streamlit as st
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.vectorstores import Cassandra
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import FlareChain
from langchain.llms import OpenAI 
from langchain.chat_models import ChatOpenAI
import uuid

# ...

# Inside your Streamlit app code
unique_key = str(uuid.uuid4())

# Initialize AstraDB connection
def get_astra():
    keyspace = 'vectordb'
    table = 'embeddings'
    cloud_config = {'secure_connect_bundle': '/Users/jauneet.singh/Downloads/secure-connect-mydb.zip'}
    auth_provider = PlainTextAuthProvider('pDTChYYSJhNXXAnvBwfuRaQR', 'banZ6KJQ0+OcuOIB6NS9t0vdtG8EpWvGgiuZ8,77Sy7D0sx+L-m8c_Aiy3vgAHe2LDBHvU6fup1zP6Q+SSSu-ozRpO2P+pZsOaARKqas,ZszybgQFYZOZgy3cM3P9Eop')
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session_astra = cluster.connect()
    return session_astra, keyspace

# Uploading the pdf with the metadata
def setup_vectorstore():
    SOURCE_DIR = "/Users/jauneet.singh/Downloads/sources"
    FILE_SUFFIX = ".pdf"

    embeddings = OpenAIEmbeddings(openai_api_key="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")

    pdf_loaders = [
        PyPDFLoader(pdf_name)
        for pdf_name in (
            f for f in (
                os.path.join(SOURCE_DIR, f2)
                for f2 in os.listdir(SOURCE_DIR)
            )
            if os.path.isfile(f)
            if f.endswith(FILE_SUFFIX)
        )
    ]

    session_astra, keyspace = get_astra()
    vectorstore = Cassandra(
        embedding=embeddings,
        session=session_astra,
        keyspace="insurance",
        table_name="vectorhealthinsurance1",
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )

    documents = [
        doc
        for loader in pdf_loaders
        for doc in loader.load_and_split(text_splitter=text_splitter)
    ]

    texts, metadatas = zip(*((doc.page_content, doc.metadata) for doc in documents))
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)

    return vectorstore

# Set up the LLM provider and embeddings
def setup_llm_and_embeddings():
    api_secret = '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'
    os.environ['OPENAI_API_KEY'] = api_secret
    openai_llm = OpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()
    return openai_llm, embeddings

# Initialize Streamlit app
st.title("💬 Chatbot")
st.caption("🚀 A chatbot powered by AstraDB")

# Initialize AstraDB session and vectorstore
session_astra, keyspace = get_astra()
vectorstore = setup_vectorstore()

# Initialize chatbot components
retriever = vectorstore.as_retriever()

openai_llm, embeddings = setup_llm_and_embeddings()

flare = FlareChain.from_llm(
    ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    retriever=retriever,
    max_generation_len=512,
    min_prob=0.3,
)

# Initialize messages if not exists
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# User input
prompt = st.text_input("User Input", key=unique_key)

# Handle user input
if prompt:
    # Append user message to messages
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant's response using FlareChain
    flare_result = flare.run(prompt)

    # Extract the assistant's response
    assistant_response = flare_result[0]["message"]["content"]

    # Append assistant's response to messages
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.text("User: " + message["content"])
    elif message["role"] == "assistant":
        st.text("Assistant: " + message["content"])
