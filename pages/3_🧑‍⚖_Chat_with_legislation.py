import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain, ChatVectorDBChain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os

os.environ["OPENAI_API_KEY"] = "" # insert api key

st.set_page_config(layout="centered", page_icon="🏛", page_title="ParliamentGPT")
st.title("📰 Chat with legislation")
st.write(
    "This app helps governments and non-profits ask questiona bout legislation and reports!"
)


def embed_doc(filename):
    if len(os.listdir("."))>0:
        loader=UnstructuredFileLoader(filename)
        raw_documents = loader.load()
        print(len(raw_documents))

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len

        )
        print("111")
        documents = text_splitter.split_documents(raw_documents)


        # Load Data to vectorstore
        embeddings = OpenAIEmbeddings()
        print("222")
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("333")


        # Save vectorstore
        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)

if os.path.exists("vectorstore.pkl"):
    with open("vectorstore.pkl","rb") as f:
        docsearch=pickle.load(f)


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the uploaded document.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about legislation and policy reports.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about the uploaded document, politely inform them that you are tuned to only answer questions about the uploaded document.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain

if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = False

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""

if "past" not in st.session_state:
    st.session_state["past"] = []

if "generated" not in st.session_state:
    st.session_state["generated"] = []
    
def process_file(uploaded_file):
    if uploaded_file is not None:
        print("yes")
    else:
        print("no")
    print("here")
    with open(uploaded_file.name,"wb") as f:
        f.write(uploaded_file.getbuffer())
        print(uploaded_file.name)
        st.write("File Uploaded successfully")
        
        with st.spinner("Document is being processed..."):
            embed_doc(uploaded_file.name)
            st.session_state["file_uploaded"] = True

            

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None and st.session_state.uploaded_file_name != uploaded_file.name:
    print("okay yes")
    print(type(st.session_state.uploaded_file_name))
    print(st.session_state.uploaded_file_name)
    st.session_state.uploaded_file_name = uploaded_file.name
    print("then")
    print(st.session_state.uploaded_file_name)
    process_file(uploaded_file)     

if "vectorstore.pkl" in os.listdir("."):
    with open("vectorstore.pkl","rb") as f:
        print("hello")
        vectorstore=pickle.load(f)
        if uploaded_file is not None and uploaded_file.name in os.listdir("."):
            os.remove(uploaded_file.name)
        
        print("Analysing document...")
    chain=get_chain(vectorstore)
else:
    print("Bye")


def get_text():
    input_text = st.text_input("You: ", value="", key="input")
    return input_text

if st.session_state["file_uploaded"]:
    user_input = get_text()
else:
    user_input = None

print(user_input)

if user_input:
    docs=vectorstore.similarity_search(user_input)
    print(len(docs))
    output = chain.run(input=user_input, vectorstore=vectorstore, context=docs[:2], chat_history=[], question=user_input, QA_PROMPT=QA_PROMPT, CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, template=_template)

    st.session_state.past.append(user_input)
    print(st.session_state.past)
    st.session_state.generated.append(output)
    print(st.session_state.past)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], avatar_style=None, key=str(i))
        message(st.session_state["past"][i], is_user=True, avatar_style=None, key=str(i) + "_user")