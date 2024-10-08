import os
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import BaseTool, Tool, tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import LanceDB
import lancedb
import pandas as pd
from langchain.chains import RetrievalQA

from lancedb.rerankers import LinearCombinationReranker
from langchain.docstore.document import Document


st.set_page_config(page_title="GlobeBotter", page_icon="bar_chart:")
# st.header('🎬 Welcome to Interview/Exam Practice Asisstant, your favourite advisor')
st.header(":bar_chart: Welcome to :blue[Interview/Exam Practice Asisstant], your favourite advisor")

load_dotenv()

#os.environ["HUGGINGFACEHUB_API_TOKEN"]
openai_api_key = os.environ['OPENAI_API_KEY']

embeddings = OpenAIEmbeddings()
uri = "data/sample-lancedb"
db = lancedb.connect(uri)

table = db.open_table('my_table')
# docsearch = LanceDB(connection = table, embedding = embeddings)

user = pd.read_csv('./data/BankChurners.csv')

# Import the movie dataset
md = pd.read_pickle('data/movies.pkl')

# ================
documents = [Document(page_content=row['text'], metadata={"source": row['weighted_rate']}) for _, row in md.iterrows()]
reranker = LinearCombinationReranker(weight=0.3)
docsearch = LanceDB.from_documents(documents, embeddings, reranker=reranker)
# ================


# Create a sidebar for user input
st.sidebar.title("_Practice Asisstant_ is :blue[cool] :sunglasses:")
st.sidebar.markdown("Please enter your details and preferences below:")

# Ask the user for age, gender and favourite movie genre
age = st.sidebar.slider("What is your age?", 1, 100, 25)
gender = st.sidebar.radio("What is your gender?", ("Male", "Female", "Other"))
# genre = st.sidebar.selectbox("What is your favourite movie genre?", md.explode('genres')["genres"].unique())
education = st.sidebar.selectbox("What is your education background?", user.explode('Education_Level')["Education_Level"].unique())


rate = st.sidebar.slider("What is your preference for grade?", 1, 100, 60)
goal = st.sidebar.radio("What is your practice goal?", ("Job Interview", "Exam"))



# Filter the movies based on the user input
# df_filtered = md[md['genres'].apply(lambda x: genre in x)]
# df_filtered = df_filtered[df_filtered['weighted_rate'].apply(lambda x: x>rate)]

df_filtered = md[md['weighted_rate'].apply(lambda x: x>rate)]


template_prefix = """You are a movie recommender system that help users to find movies that match their preferences. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}"""

user_info = """This is what we know about the user, and you can use this information to better tune your research:
Age: {age}
Gender: {gender}"""

template_suffix= """Question: {question}
Your response:"""

user_info = user_info.format(age = age, gender = gender)

COMBINED_PROMPT = template_prefix +'\n'+ user_info +'\n'+ template_suffix
print(COMBINED_PROMPT)

#setting up the chain
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'data': df_filtered}), return_source_documents=True)


query = st.text_input('Enter your question:', placeholder = 'What action movies do you suggest?')
if query:
    result = qa({"query": query})
    st.write(result['result'])

