import streamlit as st
from streamlit_chat import message
from langchain.chains import ChatVectorDBChain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import openai

openai.api_key=st.secrets["OPENAI_API_KEY"]

st.set_page_config(layout="centered", page_icon="🏛", page_title="ParliamentGPT")
st.title("🧑‍⚖ eLibrarian")
st.write(
    "eLibrarian helps you ask questions about legislation or reports. Simply upload a .pdf and eLibrarian will train a custom AI chatbot that can answer questions about it in real time. No more ctrl-F!"
)

def embed_doc(filename):
    if len(os.listdir(".")) > 0:
        loader = UnstructuredFileLoader(filename)
        raw_documents = loader.load()
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len
        )
        documents = text_splitter.split_documents(raw_documents)

        # Load Data to vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Save vectorstore to memory
        vectorstore_buffer = io.BytesIO()
        pickle.dump(vectorstore, vectorstore_buffer)
        vectorstore_buffer.seek(0)
        return vectorstore_buffer


if 'vectorstore_buffer' not in st.session_state:
    st.session_state['vectorstore_buffer'] = None


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
    with open(uploaded_file.name,"wb") as f:
        f.write(uploaded_file.getbuffer())
        print(uploaded_file.name)
        st.write("File Uploaded successfully")
        with st.spinner("Document is being processed..."):
            st.session_state['vectorstore_buffer'] = embed_doc(uploaded_file.name)
            st.session_state["file_uploaded"] = True

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None and st.session_state.uploaded_file_name != uploaded_file.name:
    st.session_state.uploaded_file_name = uploaded_file.name
    process_file(uploaded_file)     

if st.session_state['vectorstore_buffer']:
    st.session_state['vectorstore_buffer'].seek(0)
    vectorstore = pickle.load(st.session_state['vectorstore_buffer'])
    chain = get_chain(vectorstore)

def get_text():
    input_text = st.text_input("You: ", value="", key="input")
    return input_text

user_input = ""
if st.session_state["file_uploaded"]:
    user_input = get_text()

if user_input:
    docs=vectorstore.similarity_search(user_input)
    output = chain.run(input=user_input, vectorstore=vectorstore, context=docs[:2], chat_history=[], question=user_input, QA_PROMPT=QA_PROMPT, CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, template=_template)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i].strip(), avatar_style=None, key=str(i))
        message(st.session_state["past"][i].strip(), is_user=True, avatar_style=None, key=str(i) + "_user")
