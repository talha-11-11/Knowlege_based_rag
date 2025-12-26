import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chat_models import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="RAG App (VS Code)", layout="wide")
st.title("🔎 RAG System – Streamlit (VS Code)")

# -----------------------------
# Utility: Fetch website text
# -----------------------------
def fetch_text(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text("\n")
        return "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )
    except Exception as e:
        return f"Error fetching {url}: {e}"

# -----------------------------
# UI: URL input
# -----------------------------
urls = st.text_area(
    "Enter website URLs (one per line)",
    placeholder="https://example.com\nhttps://docs.company.com"
)

# -----------------------------
# Indexing
# -----------------------------
if st.button("📥 Load & Index"):
    if not urls.strip():
        st.warning("Please enter at least one URL")
    else:
        with st.spinner("Fetching and indexing data..."):
            documents = []

            for url in urls.splitlines():
                text = fetch_text(url)
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": url}
                 )
        )


            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = []
            for doc in documents:
                for c in splitter.split_text(doc.page_content):
                    chunks.append(
                        Document(
                            page_content=c,
                            metadata=doc.metadata
                        )
                    )   
            embeddings = OpenAIEmbeddings()
            vectordb = InMemoryVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            st.session_state.vectordb = vectordb
            st.success("Knowledge base created!")

# -----------------------------
# Q&A
# -----------------------------
if "vectordb" in st.session_state:
    st.subheader("💬 Ask a question")
    question = st.text_input("Your question")

    if st.button("Get Answer"):
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectordb.as_retriever(k=3)
        )

        with st.spinner("Generating answer..."):
            answer = qa_chain.run(question)

        st.markdown("### ✅ Answer")
        st.write(answer)

        st.markdown("### 📚 Sources")
        docs = st.session_state.vectordb.as_retriever(k=3)\
            .get_relevant_documents(question)

        for i, d in enumerate(docs):
            st.markdown(f"**Source {i+1}:** {d.metadata['source']}")
            st.write(d.page_content[:400] + "...")
            st.markdown("---")
