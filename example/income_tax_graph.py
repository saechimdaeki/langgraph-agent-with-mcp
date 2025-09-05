# %%
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate

embeddings= OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

vector_store= Chroma(
    embedding_function= embeddings,
    collection_name= "income_tax_collection",
    persist_directory= "./income_tax_collection"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# %%
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

graph_builder = StateGraph(AgentState)


# %%
def retrieve(state: AgentState) -> AgentState:
    """
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다.
    """
    query = state["query"]
    docs = retriever.invoke(query)
    return {"context": docs}

# %%
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0
)

# %%
from langchain_core.prompts import ChatPromptTemplate

generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers ONLY using the provided context. "
            "If the context does not contain the answer, reply with '모르겠습니다'. "
            "Answer in Korean. Do NOT include chain-of-thought or <think> tags.",
        ),
        (
            "user",
            "질문: {question}\n\n"
            "컨텍스트(발췌):\n{context}\n\n"
            "규칙:\n- 근거가 없으면 '모르겠습니다'.\n- 핵심 bullet 후 짧은 결론.",
        ),
    ]
)

def _join_docs_for_prompt(docs: List[Document], max_chars: int = 6000) -> str:
    parts, used = [], 0
    for d in docs:
        piece = (d.page_content or "")[:1200]
        if used + len(piece) > max_chars:
            break
        parts.append(f"- {piece}")
        used += len(piece)
    return "\n".join(parts) if parts else "(no context)"

def generate(state: AgentState) -> AgentState:
    context_docs = state["context"]
    query = state["query"]
    ctx_str = _join_docs_for_prompt(context_docs)

    rag_chain = generate_prompt | llm
    response = rag_chain.invoke({"question": query, "context": ctx_str})
    content = response.content if hasattr(response, "content") else str(response)
    return {"answer": content}


# %%
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate

doc_relevance_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict classifier that decides if the documents are relevant to the question.\n"
            'Respond ONLY with JSON: {{"Score": 1}} if relevant, {{"Score": 0}} if irrelevant.\n'
            "Do NOT include chain-of-thought or <think>.",
        ),
        ("user", "Question:\n{question}\n\nDocuments:\n{documents}"),
    ]
)

def _as_text(docs: List[Document], max_chars: int = 6000) -> str:
    parts, used = [], 0
    for d in docs:
        t = (d.page_content or "")[:1200]
        if used + len(t) > max_chars:
            break
        parts.append(t)
        used += len(t)
    return "\n---\n".join(parts)

def check_doc_relevance(state: AgentState) -> Literal["relevant", "irrelevant"]:
    query = state["query"]
    context_docs = state["context"]
    docs_str = _as_text(context_docs)

    chain = doc_relevance_prompt | llm
    resp = chain.invoke({"question": query, "documents": docs_str})
    text = resp.content if hasattr(resp, "content") else str(resp)
    score = 1 if '"Score": 1' in text or "'Score': 1" in text or "Score\": 1" in text else 0
    return "relevant" if score == 1 else "irrelevant"


# %%
dictionary = ["사람과 관련된 표현 -> 거주자"]

rewrite_prompt = PromptTemplate.from_template(
    f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요 
사전: {dictionary}
질문: {{query}}
"""
)

# %%
from langchain_core.output_parsers import StrOutputParser

def rewrite(state: AgentState) -> AgentState:
    query = state["query"]
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    new_query = rewrite_chain.invoke({"query": query})
    return {"query": new_query}

# %%
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict grader. Decide if the student's answer is grounded in the provided documents.\n"
            'Return ONLY JSON: {{"Score": 1}} if hallucinated (NOT supported), {{"Score": 0}} if not hallucinated.\n'
            "Do NOT include chain-of-thought or <think>.",
        ),
        ("user", "documents:\n{documents}\n\nstudent_answer:\n{student_answer}"),
    ]
)


def check_hallucination(state: AgentState) -> Literal["hallucinated", "not hallucinated"]:
    answer = state["answer"]
    docs = state["context"]
    docs_str = _as_text(docs)

    chain = hallucination_prompt | llm
    resp = chain.invoke({"student_answer": answer, "documents": docs_str})
    text = resp.content if hasattr(resp, "content") else str(resp)
    score = 1 if '"Score": 1' in text or "'Score': 1" in text or "Score\": 1" in text else 0
    return "hallucinated" if score == 1 else "not hallucinated"

# %%

helpfulness_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpfulness grader. Judge if the answer is useful for the user question.\n"
            'Return ONLY JSON: {{"Score": 1}} if helpful, {{"Score": 0}} if unhelpful.\n'
            "Do NOT include chain-of-thought or <think>.",
        ),
        ("user", "Question:\n{question}\n\nAnswer:\n{student_answer}"),
    ]
)

def check_helpfulness_grader(state: AgentState) -> str:
    query = state["query"]
    answer = state["answer"]
    chain = helpfulness_prompt | llm
    resp = chain.invoke({"question": query, "student_answer": answer})
    text = resp.content if hasattr(resp, "content") else str(resp)
    score = 1 if '"Score": 1' in text or "'Score': 1" in text or "Score\": 1" in text else 0
    return "helpful" if score == 1 else "unhelpful"

def check_helpfulness(state: AgentState) -> AgentState:
    return state

# %%
from langgraph.graph import START, END


graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("rewrite", rewrite)
graph_builder.add_node("check_helpfulness", check_helpfulness)

graph_builder.add_edge(START, "retrieve")

graph_builder.add_conditional_edges(
    "retrieve",
    check_doc_relevance,
    {
        "relevant": "generate",
        "irrelevant": END,
    },
)

graph_builder.add_conditional_edges(
    "generate",
    check_hallucination,
    {
        "not hallucinated": "check_helpfulness",
        "hallucinated": "generate",
    },
)

graph_builder.add_conditional_edges(
    "check_helpfulness",
    check_helpfulness_grader,
    {
        "helpful": END,
        "unhelpful": "rewrite",
    },
)

graph_builder.add_edge("rewrite", "retrieve")

# %%
graph = graph_builder.compile()





