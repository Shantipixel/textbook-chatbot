import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
import tiktoken
import boto3
import os
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Deep Learning Textbook Chatbot", page_icon="📚")
st.title("📚 Deep Learning Textbook Chatbot")
with st.sidebar:
    model_choice = st.radio("Choose AI Model:", ["Groq (Llama 3.3)", "Bedrock"])
# ── Load resources once ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="./deeplearningchromadb")
    return client.get_collection("deep_learning_book")

@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.2
    )

@st.cache_resource
def load_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

@st.cache_resource
def load_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=st.secrets["AWS_DEFAULT_REGION"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )
bedrock_client = load_bedrock_client()
def bedrock_generate(prompt):
    response = bedrock_client.converse(
        modelId="us.anthropic.claude-4-5-20250929-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        inferenceConfig={"temperature": 0.2}
    )
    return response["output"]["message"]["content"][0]["text"]

model = SentenceTransformer("all-MiniLM-L6-v2")
collection = load_collection()
llm = load_llm()
tokenizer = load_tokenizer()

# ── Helper functions ──────────────────────────────────────────────────────────
def truncate(text, max_tokens=400):
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])
def retrieve_chunks(question, n_results=5):
    question_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results

def build_prompt(question, results, conversation_history):
    context = "\n\n".join([
        f"[Chapter: {meta['chapter_title']} | Page: {meta['page']}]\n{truncate(doc)}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ])
    history_text = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in conversation_history[-6:]  # last 3 turns
    ])
    return f"""You are a helpful assistant for the textbook 'Deep Learning' by Ian Goodfellow.
Answer the question based on the context provided below.
If the context is relevant but does not contain an explicit definition,
synthesize an answer from the available information.
If the context is completely unrelated, say 'I could not find this in the textbook.'
Context:
{context}
Summary of earlier conversation:
{st.session_state.chat_summary}
Recent Conversation History:
{history_text}
Question: {question}
Answer:"""
def update_summary(old_messages, existing_summary):
    text_to_summarize = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in old_messages
    ])
    summarization_prompt = f"""Summarize the following conversation in 1 to 5 concise lines,
capturing only the key facts, questions, and topics discussed. Merge it with the existing summary
if relevant, avoiding repetition.
Existing summary:
{existing_summary}
New conversation to incorporate:
{text_to_summarize}
Updated summary:"""
    response = llm.invoke(summarization_prompt)
    return response.content

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_summary" not in st.session_state:
    st.session_state.chat_summary = ""
# ── Display chat history ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about Deep Learning..."):
    # display user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    # retrieve and generate
    with st.chat_message("assistant"):
        with st.spinner("Searching textbook..."):
            results = retrieve_chunks(question)
            prompt = build_prompt(question, results, st.session_state.messages)
            if model_choice == "Groq (Llama 3.3)":
                response = llm.invoke(prompt)
                answer = response.content
            else:
                answer = bedrock_generate(prompt)
        st.markdown(answer)
        # show sources in expander
        with st.expander("📖 Sources"):
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                st.markdown(f"**Chapter:** {meta['chapter_title']} | **Page:** {meta['page']} | **Score:** {1-dist:.2f}")
                st.markdown(doc[:200] + "...")
                st.divider()
    st.session_state.messages.append({"role": "assistant", "content": answer})
    if len(st.session_state.messages) > 6:
        messages_to_summarize = st.session_state.messages[:-6]
        st.session_state.chat_summary = update_summary(
            messages_to_summarize, st.session_state.chat_summary
        )
        st.session_state.messages = st.session_state.messages[-6:]