import os
import openai
import streamlit as st
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Set up OpenAI API key
# os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
openai.api_key = 'Your_OPENAI_API_KEY' # os.getenv('OPENAI_API_KEY')

# Initialize the GPT model
llm_model = "gpt-4o-mini"
llm = ChatOpenAI(model_name=llm_model, temperature=0.2)

# Set up Chroma database for data storage and retrieval
persist_directory = '/db'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Response template
template = """Use the following pieces of context to answer the questions at the end.
If you don't know the answer, just say you don't know, don't try to make up the answer.
Keep your answer as concise as possible.
Always say at the start "I am AI-based WIKI! Thanks for your question!".
{context}
Question: {question}
Answer: """
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# Initialize memory to store conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize retrieval chain for answer generation
retriever = vectordb.as_retriever(k=3)
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT}    
)

# Function to get the answer from the bot
def get_answer(question):
    result = qa.invoke({"question": question})
    return result['answer']


st.title("Wiki AI Chatbot powered by OpenAI and Langchain")

user_question = st.text_input("Enter your question:")

# Chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Send button to submit the question
if st.button("Send"):
    if user_question:
        # Append user question and get the answer
        st.session_state['chat_history'].append({"role": "user", "content": user_question})
        answer = get_answer(user_question)
        st.session_state['chat_history'].append({"role": "assistant", "content": answer})

# Display chat history
for message in st.session_state['chat_history']:
    if message['role'] == 'user':
        st.write(f"**You**: {message['content']}")
    else:
        st.write(f"**Wiki Assistant**: {message['content']}")

# Clear button to reset chat history
if st.button("Clear"):
    st.session_state['chat_history'] = []
    memory.clear()
    st.success("Chat history cleared. Start a new conversation.")
