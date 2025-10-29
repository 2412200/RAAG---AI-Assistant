from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
    task="text-generation"
)
model = ChatHuggingFace(llm=endpoint)
parser = StrOutputParser()
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_TOKEN")

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

prompt = PromptTemplate(
    """Anser the question"""
)

chain = prompt | model
chain.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})
result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')

st.set_page_config(page_title="RAAG - AI Code Generator", layout="wide")
st.title("RAAG - AI Assistant")
task = st.text_area("Enter your project idea or description:")
generate_button = st.button("🚀 Submit")

if generate_button:
    if not task.strip():
        st.warning("⚠️ Please describe your project first.")
    else:
        col1, col2 = st.columns(2)

        with st.spinner("Generating frontend..."):
            frontend_result = front_chain.invoke({"task": task})

        with st.spinner("Generating backend..."):
            backend_result = back_chain.invoke({"task": task})

        with col1:
            st.subheader("🧩 Frontend Code (React)")
            st.code(frontend_result, language="javascript")

        with col2:
            st.subheader("⚙️ Backend Code (API + Logic)")
            st.code(backend_result, language="python")

