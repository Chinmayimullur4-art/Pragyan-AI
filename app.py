import os
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq


# -----------------------------
# 🔐 ENV CONFIG
# -----------------------------
GROQ_API_KEY =setx GROQ_API_KEY "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"

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not set in environment variables")
    st.stop()


# -----------------------------
# 📊 CREATE DEFAULT FAQ FILE
# -----------------------------
def create_default_excel():
    if not os.path.exists("pragyan_faq_prices.xlsx"):
        faq_data = {
            "Category": ["Program Overview"],
            "Question": ["What is the program duration?"],
            "Answer": ["18 months (6 months training + 12 months internship)"]
        }
        df = pd.DataFrame(faq_data)
        df.to_excel("pragyan_faq_prices.xlsx", index=False)

create_default_excel()


# -----------------------------
# 🧠 PROMPTS
# -----------------------------
SALES_PROMPTS = {
    "Student Counselor": """You are Aarav, a PragyanAI counselor.

Context:
{context}

Answer clearly and persuasively."""
}


# -----------------------------
# 📦 VECTOR STORE
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = None


def load_documents(files=None):
    global vectorstore
    docs = []

    if files:
        for file in files:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file.name)
                docs.extend(loader.load())

            elif file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file)
                for _, row in df.iterrows():
                    docs.append(Document(page_content=str(row.to_dict())))

    # Load default file
    df = pd.read_excel("pragyan_faq_prices.xlsx")
    for _, row in df.iterrows():
        docs.append(Document(page_content=str(row.to_dict())))

    vectorstore = FAISS.from_documents(docs, embeddings)


load_documents()


# -----------------------------
# 🤖 LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)


# -----------------------------
# 💬 MEMORY
# -----------------------------
store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# -----------------------------
# 🔗 RAG CHAIN
# -----------------------------
def create_chain(context):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SALES_PROMPTS["Student Counselor"].format(context=context)),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])
    return prompt | llm | StrOutputParser()


def respond(message):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(message)

    context = "\n".join([d.page_content for d in docs])

    chain = create_chain(context)

    convo = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    return convo.invoke(
        {"input": message},
        config={"configurable": {"session_id": "default"}}
    )


# -----------------------------
# 🌐 STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="PragyanAI Assistant", layout="wide")

st.title("🤖 PragyanAI Assistant")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs / Excel",
    type=["pdf", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    load_documents(uploaded_files)
    st.sidebar.success("✅ Knowledge Base Updated")


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    response = respond(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)


if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    store.clear()
    st.sidebar.success("Cleared!")
