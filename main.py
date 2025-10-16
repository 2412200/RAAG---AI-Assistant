import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from loadenv import load_env

load_env()
hf_token = os.getenv("HUGGING_FACE_TOKEN")

st.set_page_config(page_title="RAAG - AI Code Generator", layout="wide")
st.title("RAAG - AI Assistant")

endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
    task="text-generation"
)
model = ChatHuggingFace(llm=endpoint)
parser = StrOutputParser()

task = st.text_area("Enter your project idea or description:")
generate_button = st.button("🚀 Submit")

frontend_template = PromptTemplate(
    template="""
    You are an expert React developer.
    Based on the project task below:
    {task}
    write a code for React-based frontend UI structure (components, routes, and styling).
    Make it responsive and user-friendly.
    """,
    input_variables=["task"]
)
frontend_chain = frontend_template | model | parser

backend_template = PromptTemplate(
    template="""
    You are an experienced backend developer.
    Based on the project task below:
    {task}
    Create backend logic with REST or GraphQL APIs, authentication, and database schema.
    Prefer Python (FastAPI) or Node.js (Express) with clean, production-ready code.
    """,
    input_variables=["task"]
)
backend_chain = backend_template | model | parser

if generate_button:
    if not task.strip():
        st.warning("⚠️ Please describe your project first.")
    else:
        col1, col2 = st.columns(2)

        with st.spinner("Generating frontend..."):
            frontend_result = frontend_chain.invoke({"task": task})

        with st.spinner("Generating backend..."):
            backend_result = backend_chain.invoke({"task": task})

        with col1:
            st.subheader("🧩 Frontend Code (React)")
            st.code(frontend_result, language="javascript")

        with col2:
            st.subheader("⚙️ Backend Code (API + Logic)")
            st.code(backend_result, language="python")


