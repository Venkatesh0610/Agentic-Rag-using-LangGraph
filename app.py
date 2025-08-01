# === Imports and Setup ===
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, Sequence, TypedDict, Literal
from PIL import Image
import os
import warnings

# LangGraph and LangChain components
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")
load_dotenv()

# === FastAPI Setup ===
app = FastAPI()

# Enable CORS for local/frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load LLM and Embeddings ===
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="Gemma2-9b-It", temperature=0, streaming=True)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# RAG Implementation
# === Load and Process Document ===
CACHE_FILE = "cached_blog.txt"

# If already cached, read the file
if Path(CACHE_FILE).exists():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        raw_content = f.read()
else:
    # Otherwise, load from the web
    docs = WebBaseLoader("https://medium.com/towards-artificial-intelligence/ai-agents-explained-theory-applications-and-python-implementation-5de81ce7cd92").load()
    raw_content = docs[0].page_content
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(raw_content)

# === Split document into chunks for vector search ===
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=25)
doc_splits = text_splitter.split_documents([Document(page_content=raw_content)])

# === Vector Store Setup with FAISS ===
VECTOR_DIR = "faiss_index"
if os.path.exists(f"{VECTOR_DIR}/index.faiss"):
    vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(doc_splits, embedding=embeddings)
    vectorstore.save_local(VECTOR_DIR)

# === Create Retrieval Tool for LangGraph ===
retriever = vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "this is related ai agents blogs",
)


tools = [retriever_tool]

# === Define LangGraph State ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# === Grading Node for Document Relevance ===
class GradeOutput(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")

# Determine whether to use retrieved context or not
def grade_documents(state) -> Literal["generate", "rewrite"]:
    print("=== [NODE: RETRIEVE] ===")
    model = ChatGroq(temperature=0, model="llama-3.3-70b-versatile").with_structured_output(GradeOutput)
    messages = state["messages"]
    question = messages[0].content
    context = messages[-1].content

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
                    Document:
                    {context}
                    Question: {question}
                    Is the document relevant? Answer 'yes' or 'no'.""",
                            input_variables=["context", "question"]
                        )

    result = (prompt | model).invoke({"context": context, "question": question})
    if result.binary_score == "yes":
        print("=== [DECISION: DOCS RELEVANT] ===")
        return "generate"
    else:
        print("=== [DECISION: DOCS NOT RELEVANT] ===")
        return "rewrite"

# === Main Agent Node ===
def agent(state):
    print("=== [NODE: AGENT] ===")
    messages = state["messages"]
    llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

    # If follow-up, use concise format
    if len(messages) > 1:
        last_message = messages[-1]
        question = last_message.content

        prompt = PromptTemplate(
            template="""You are a concise assistant. 
                        Only answer the question directly in one sentence or less. 
                        Do NOT explain, expand, reflect, or rephrase.
                        Make sure to give little details about the question, don't answer in single line.
                        Question: {question}""",
            input_variables=["question"]
        )

        chain = prompt | llm
        response = chain.invoke({"question": question})
        return {"messages": [AIMessage(content=response.content)]}
    else:
        # First-time user message, allow tools
        llm_with_tool = llm.bind_tools(tools)
        response = llm_with_tool.invoke(messages)
        return {"messages": [response]}

# === Rewrite Node to Improve User's Question ===
def rewrite(state):
    print("=== [NODE: REWRITE] ===")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f"""\n 
                    Look at the input and try to reason about the underlying semantic intent / meaning and make the question more detailed. \n 
                    Here is the initial question:
                    \n ------- \n
                    {question} 
                    \n ------- \n
                    Formulate an improved question: """,
                    )
    ]
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, streaming=True)
    return {"messages": [model.invoke(msg)]}

# === Generate Answer from Context ===
def generate(state):
    print("=== [NODE: GENERATE] ===")
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
                    You are a helpful assistant. Use the context below to answer the question.
                    
                    Context:
                    ---------
                    {context}
                    ---------
                    
                    Question: {question}
                    Answer:
                    """
                            )

    llm_gen = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, streaming=True)
    response = (prompt | llm_gen | StrOutputParser()).invoke({"context": docs, "question": question})
    return {"messages": [response]}

# === LangGraph Flow Definition ===
workflow = StateGraph(AgentState)

# Define Nodes
workflow.add_node("agent", agent)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

# Define Edges / Flow Logic
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile the graph
graph = workflow.compile()

# === Visualize the LangGraph DAG (only once) ===
if not Path("visualization.png").exists():
    image_data = graph.get_graph().draw_mermaid_png()
    with open("visualization.png", "wb") as f:
        f.write(image_data)

# === API Schemas ===
class Query(BaseModel):
    message: str

# === FastAPI Endpoints ===

# Serve HTML UI
@app.get("/")
async def serve_html():
    return FileResponse("chatbot_ui.html")

# Main chat endpoint
@app.post("/chat")
async def chat_endpoint(query: Query):
    try:
        print("******** FLOW ********")
        print("=== [START] ===")
        # Run LangGraph pipeline
        events = graph.stream({"messages": [HumanMessage(content=query.message)]}, stream_mode="values")
        response = ""
        for event in events:
            response = event["messages"][-1].content
        print("=== [END] ===")
        return JSONResponse(content={"response": response})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
