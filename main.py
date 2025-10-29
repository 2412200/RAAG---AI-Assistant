import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict

# -----------------------------
# Environment setup
# -----------------------------
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_TOKEN")

# -----------------------------
# Model setup
# -----------------------------
endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)

model = ChatHuggingFace(llm=endpoint)
parser = StrOutputParser()

# -----------------------------
# State Definition
# -----------------------------
class BotState(TypedDict):
    task: str
    result: str
    response: str
    frontend_result: str
    backend_result: str


# -----------------------------
# Function Definitions
# -----------------------------
def classify(state: BotState) -> BotState:
    classify_prompt = PromptTemplate(
        template="Classify the following task as either 'Code' or 'Conv': {task}",
        input_variables=["task"]
    )
    classify_chain = classify_prompt | model | parser
    result = classify_chain.invoke({"task": state["task"]}).strip()
    state["result"] = "Code" if "code" in result.lower() else "Conv"
    return state


def frontend(state: BotState) -> BotState:
    frontend_template = PromptTemplate(
        template="""
        You are an expert React developer.
        Based on the project task below:
        {task}
        Write code for a React-based frontend UI structure (components, routes, styling).
        Make it responsive and modern.
        """,
        input_variables=["task"]
    )
    chain = frontend_template | model | parser
    state["frontend_result"] = chain.invoke({"task": state["task"]})
    return state


def backend(state: BotState) -> BotState:
    backend_template = PromptTemplate(
        template="""
        You are an experienced backend developer.
        Based on the project task below:
        {task}
        Create backend logic with REST or GraphQL APIs, authentication, and database schema.
        Prefer Python (FastAPI) or Node.js (Express).
        """,
        input_variables=["task"]
    )
    chain = backend_template | model | parser
    state["backend_result"] = chain.invoke({"task": state["task"]})
    return state


def code(state: BotState) -> BotState:
    parallel = RunnableParallel(frontend=frontend, backend=backend)
    results = parallel.invoke(state)
    state.update(results)
    return state


def conv(state: BotState) -> BotState:
    conv_template = PromptTemplate(
        template="You are a helpful AI assistant. Answer politely to: {task}",
        input_variables=["task"]
    )
    chain = conv_template | model | parser
    state["response"] = chain.invoke({"task": state["task"]})
    return state


# -----------------------------
# Build Graph
# -----------------------------
graph = StateGraph(BotState)

graph.add_node("classify", classify)
graph.add_node("code", code)
graph.add_node("conv", conv)

graph.add_edge(START, "classify")

def branch(state: BotState) -> str:
    return state["result"]

graph.add_conditional_edges(
    "classify",
    branch,
    {"Code": "code", "Conv": "conv"}
)

graph.add_edge("code", END)
graph.add_edge("conv", END)

workflow = graph.compile()


# -----------------------------
# Streamlit Frontend
# -----------------------------
st.set_page_config(page_title="AI Project Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Project Assistant")
st.write("Enter a project idea or query, and the AI will either build your app structure or answer conversationally.")

task = st.text_area("Enter your task or idea", placeholder="e.g., Create a gym website or What is LangChain?")

if st.button("Run Workflow"):
    if not task.strip():
        st.warning("Please enter a task first.")
    else:
        with st.spinner("Processing your request..."):
            result = workflow.invoke({"task": task})
        
        st.success("✅ Task processed successfully!")

        # Display results dynamically
        if result.get("result") == "Code":
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🧩 Frontend Code (React)")
                st.code(result.get("frontend_result", "No frontend output"), language="javascript")

            with col2:
                st.subheader("⚙️ Backend Code (API)")
                st.code(result.get("backend_result", "No backend output"), language="python")

        else:
            st.subheader("💬 AI Response")
            st.write(result.get("response", "No response generated."))
