import streamlit as st
from streamlit_chat import message
from langchain.chains import ChatVectorDBChain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

openai.api_key=st.secrets["OPENAI_API_KEY"]

st.set_page_config(layout="centered", page_icon="🏛", page_title="ParliamentGPT")
st.title("🧑‍⚖ eLibrarian")
st.write(
    "eLibrarian helps you ask questions about legislation or reports. Simply upload a .pdf and eLibrarian will train a custom AI chatbot that can answer questions about it in real time. No more ctrl-F!"
)

if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = False

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""

if "past" not in st.session_state:
    st.session_state["past"] = []

if "generated" not in st.session_state:
    st.session_state["generated"] = []

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

def embed_doc(filename):
    loader = UnstructuredFileLoader(filename)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
        chunk_overlap=0,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    return vectorstore

def get_chain(vectorstore):
    llm = OpenAI(temperature=0)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain
    
def process_file(uploaded_file):
    with open(uploaded_file.name,"wb") as f:
        f.write(uploaded_file.getbuffer())
        print(uploaded_file.name)
        st.write("File Uploaded successfully")
        
        with st.spinner("Document is being processed..."):
            embed_doc(uploaded_file.name)
            st.session_state["file_uploaded"] = True
    return uploaded_file.name  # Add this line


def get_text():
    input_text = st.text_input("Prompt: ", value="", key="input")
    return input_text if input_text is not None else ""


uploaded_file = st.file_uploader("Upload your file:")

if uploaded_file is not None: 
    file = process_file(uploaded_file)
    vectorstore = embed_doc(file)
    chain = get_chain(vectorstore)

    user_input = get_text()

    if user_input and st.session_state["file_uploaded"]:
        with st.spinner("Drafting response..."):
            docs=vectorstore.similarity_search(user_input)
            output = chain.run(input=user_input, vectorstore=vectorstore, context=docs[:2], chat_history=[], question=user_input, QA_PROMPT=QA_PROMPT, CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, template=_template).strip()
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], avatar_style=None, key=str(i))
            message(st.session_state["past"][i], is_user=True, avatar_style=None, key=str(i) + "_user")
