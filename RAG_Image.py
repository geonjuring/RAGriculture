from dotenv import load_dotenv
load_dotenv()

# from rag.pdf import PDFRetrievalChain
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain_teddynote.document_loaders import HWPLoader
from typing import Literal, List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
# from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.tools.tavily import TavilySearch
from typing_extensions import TypedDict, Annotated
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_teddynote.messages import stream_graph
from langchain_core.runnables import RunnableConfig
import uuid
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor




# 웹 검색 도구 생성
web_search_tool = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"), max_results=3)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# 이미지 분석을 위한 OpenAI Vision 모델 초기화
vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)

file_path = [r"Data/hwp/딸기(촉성재배) 농작업일정.hwp",
             r"Data/hwp/토마토,방울토마토 농작업일정.hwp",
             r"Data/pdf/과즙이풍부한왕딸기킹스베리재배매뉴얼.pdf",
             r"Data/pdf/과채류(토마토).pdf",
             r"Data/pdf/29 토마토_저화질_단면.pdf",
             r"Data/pdf/2024 딸기 재배기술정보(병해충).pdf",
             r"Data/pdf/겨울철 주요 병해충 관리기술.pdf",
             r"Data/pdf/여름병충해발생대비,딸기모종관리기술.pdf",
             r"Data/pdf/토마토 농약.pdf",
             r"Data/pdf/딸기 재배 기술 정론(최신 교정).pdf",
             r"Data/pdf/딸기 재배 기술 총람 (최신 교정).pdf",
             r"Data/pdf/딸기 재배 일정 통일 및 상세화.pdf",
             r"Data/pdf/토마토 반촉성재배 기술 정론 (교정).pdf",
             r"Data/pdf/토마토 반촉성재배 상세 일정 생성.pdf",
             r"Data/pdf/토마토 상업 재배 기술 총람 (교정).pdf",
             ]

def load_docs(paths: List[str]) -> List:
    all_docs = []
    
    for path in paths:
        if path.lower().endswith(".pdf"):
            loader = PyMuPDFLoader(path)
            # loader = UnstructuredPDFLoader(path)
        elif path.lower().endswith(".hwp"):
            loader = HWPLoader(path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {path}")
        docs = loader.load()
        
        for doc in docs:
            split_docs = text_splitter.split_text(doc.page_content)  # 문서 텍스트 분할
            all_docs.extend(split_docs)

    return all_docs

file = load_docs(file_path)

# 최신 LLM 모델 이름 가져오기
MODEL_NAME = "gpt-4o-mini"



extended_model = AutoModelForImageClassification.from_pretrained("./ML_Model")
extended_processor = AutoImageProcessor.from_pretrained("./ML_Model")
    
    # 클래스 매핑 로드 (class_mapping.txt에서)
all_classes = {}
with open("./ML_Model/class_mapping.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # 클래스 매핑 파싱 (숫자: 클래스명 형식)
        if ":" in line and not line.startswith("=") and not line.startswith("클래스"):
            try:
                idx, name = line.split(": ", 1)
                all_classes[int(idx)] = name
            except ValueError:
                continue  # 파싱 실패 시 무시




# 사용자 쿼리를 가장 관련성 높은 데이터 소스로, 이미지가 있다면 이미지 분석 노드로 라우팅하는 데이터 모델
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource, or image analysis."""

    # 데이터 소스 선택을 위한 리터럴 타입 필드
    datasource: Literal["vectorstore", "web_search", "image_analysis"] = Field(
        ...,
        description="Given a user question choose to route it to web search, vectorstore, or image analysis.",
    )


# LLM 초기화 및 함수 호출을 통한 구조화된 출력 생성
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)


# 시스템 메시지와 사용자 질문을 포함한 프롬프트 템플릿 생성
system = """You are an expert at routing a user question to a vectorstore, web search, or image analysis.
The vectorstore contains documents related to Crop cultivation, pest diagnosis, pesticide and fertilizer(compost)
Use the vectorstore for questions on these topics.
Otherwise, use web-search.
If the user provides a plant image, use image analysis."""

# Routing 을 위한 프롬프트 템플릿 생성
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 프롬프트 템플릿과 구조화된 LLM 라우터를 결합하여 질문 라우터 생성
question_router = route_prompt | structured_llm_router


# 이미지 분석 결과를 가장 관련성 높은 데이터 소스로 라우팅하는 데이터 모델 
class ImageRouteQuery(BaseModel):
    """Route a user question to the most relevant datasource after image analysis."""

    # 데이터 소스 선택을 위한 리터럴 타입 필드
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search, vectorstore.",
    )


# LLM 초기화 및 함수 호출을 통한 구조화된 출력 생성
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
structured_image_llm_router = llm.with_structured_output(ImageRouteQuery)




# The key principle: Use vectorstore when you have relevant documents, otherwise use web-search."""
system = """You are an expert at routing a user question to a vectorstore, web search
The vectorstore contains documents related to Crop cultivation, pest diagnosis, pesticide and fertilizer(compost)
Use the vectorstore for questions on these topics.
Otherwise, use web-search."""


# Routing 을 위한 프롬프트 템플릿 생성
image_route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 프롬프트 템플릿과 구조화된 LLM 라우터를 결합하여 질문 라우터 생성
image_route_router = image_route_prompt | structured_image_llm_router




# 문서 평가를 위한 데이터 모델 정의
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM 초기화 및 함수 호출을 통한 구조화된 출력 생성
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 시스템 메시지와 사용자 질문을 포함한 프롬프트 템플릿 생성
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# 문서 검색결과 평가기 생성
retrieval_grader = grade_prompt | structured_llm_grader



# LangChain Hub에서 프롬프트 가져오기(RAG 프롬프트는 자유롭게 수정 가능)
# prompt = hub.pull("teddynote/rag-prompt")


system = """You are an AI assistant specializing in Question-Answering (QA) tasks within a Retrieval-Augmented Generation (RAG) system. 
Your primary mission is to answer questions based on provided context or chat history.
Ensure your response is concise and directly addresses the question without any additional narration.

###

Your final answer should be written concisely (but include important numerical values, technical terms, jargon, and names), followed by the source of the information.

# Steps

1. Carefully read and understand the context provided.
2. Identify the key information related to the question within the context.
3. Formulate a concise answer based on the relevant information.
4. Ensure your final answer directly addresses the question.
5. List the source of the answer in bullet points, which must be a file name (with a page number) or URL from the context. Omit if the source cannot be found.

# Output Format:
❓ 질문 범주 및 출력 형식:

📌 병해충에 대한 질문일 경우 혹은 이미지 분석 결과가 병해충명인 경우:
✅ 병해충 이름: [정확한 병해충명(한국명칭)]
🧴 추천 농약: [구체적인 농약명]
💊 농약 사용 방법: [희석비율, 살포량, 살포방법 등 구체적 방법]
🕓 농약 사용 시기: [구체적인 시기와 주기]
🧰 방제 방법: [예방법과 치료법 구체적으로]

📌 농약에 대한 질문일 경우:
🧴 추천 농약: [구체적인 농약명]
💊 농약 사용 방법: [희석비율, 살포량, 살포방법 등 구체적 방법]
🕓 농약 사용 시기: [구체적인 시기와 주기]

📌 비료나 퇴비에 대한 질문일 경우:
🌿 필요한 영양소: [구체적인 영양소명과 함량]
📆 비료 사용 시기: [구체적인 시기와 주기]
🧪 비료 사용 방법: [구체적인 시비량과 방법]
⚖️ 비료 사용 용량: [구체적인 양과 단위]

📌 작물 재배법이 궁금한 경우 혹은 이미지 분석 결과가 작물이 건강한 경우:
🌱 재배 시기: [구체적인 파종/정식 시기]
🌍 재배 환경 조건: [구체적인 온도, 습도, 토양조건]
📏 재식 간격 및 정식 방법: [구체적인 간격과 방법]
💧 관수 방법: [구체적인 관수량과 주기]
🧪 시비(비료) 방법: [구체적인 시비량과 방법]
✂️ 생육 관리: [구체적인 관리방법]
🌾 수확 시기 및 방법: [구체적인 수확시기와 방법]

**Source**(Optional)
- (Source of the answer, must be a file name(with a page number) or URL from the context. Omit if you can't find the source of the answer.)
- (list more if there are multiple sources)
- ...

###

Remember:
- It's crucial to base your answer solely on the **PROVIDED CONTEXT**. 
- DO NOT use any external knowledge or information not present in the given materials.
- If you can't find the source of the answer, you should answer that you don't know.
- When using the Web Search Tool, prioritize searching from official agricultural sources such as:
  * 농촌진흥청 (Rural Development Administration) - https://www.rda.go.kr
  * 스마트팜코리아 (Smart Farm Korea) - https://www.smartfarmkorea.net
- If it is based on a web search, make sure to add the following phrase at the bottom of the result:
  🔎 웹 검색 결과를 기반으로 제공된 정보입니다.
-💡 Explain it in a concise and easy language so that even novice farmers can understand it.
- Specific figures and methods: Provide specific figures such as dilution ratio, injection amount, timing, etc.
- Step by Step: Describe the complex process step by step.
###

# Here is the user's QUESTION that you should answer:
{question}

# Here is the CONTEXT that you should use to answer the question:
{context}

# Your final ANSWER to the user's QUESTION:
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}, LLM generation:{context}")
    ]
)



# LLM 초기화
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)


# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(
        [
            f'<document><content>{doc.page_content}</content><source>{doc.metadata["source"]}</source><page>{doc.metadata["page"]+1}</page></document>'
            for doc in docs
        ]
    )


# RAG 체인 생성
rag_chain = prompt | llm | StrOutputParser()


# 할루시네이션 체크를 위한 데이터 모델 정의
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# 함수 호출을 통한 LLM 초기화
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 프롬프트 설정
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# 프롬프트 템플릿 생성
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# 환각 평가기 생성
hallucination_grader = hallucination_prompt | structured_llm_grader

class GradeAnswer(BaseModel):
    """Binary scoring to evaluate the appropriateness of answers to questions"""

    binary_score: str = Field(
        description="Indicate 'yes' or 'no' whether the answer solves the question"
    )


# 함수 호출을 통한 LLM 초기화
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# 프롬프트 설정
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

# 프롬프트 템플릿과 구조화된 LLM 평가기를 결합하여 답변 평가기 생성
answer_grader = answer_prompt | structured_llm_grader

# LLM 초기화
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# Query Rewriter 프롬프트 정의(자유롭게 수정이 가능합니다)
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

# Query Rewriter 프롬프트 템플릿 생성
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

# Query Rewriter 생성
question_rewriter = re_write_prompt | llm | StrOutputParser()

# 그래프의 상태 정의
class GraphState(TypedDict):
    """
    그래프의 상태를 나타내는 데이터 모델

    Attributes:
        question: 질문
        generation: LLM 생성된 답변
        documents: 도큐먼틑 리스트
        image : 이미지 파일
        image_result : 이미지 분석 결과
    """

    question: Annotated[str, "User question"]
    generation: Annotated[str, "LLM generated answer"]
    documents: Annotated[List[str], "List of documents"]
    image : Annotated[str, "Image file"]
    image_result : Annotated[str, "Image analysis result"]
    

def some_keyword_matching(doc, question):
    return any(keyword in doc for keyword in question.split())


# 문서 검색 노드
def retrieve(state):
    # print("==== [RETRIEVE] ====")
    question = state["question"]
    image_result = state.get("image_result", "")

    if image_result:
        enhanced_question = f"{question} (병해충명 : {image_result})"
    else:
        enhanced_question = question
    

    documents = [doc for doc in file if some_keyword_matching(doc, enhanced_question)]

    # print(f"검색된 문서 수: {len(documents)}")
    # print(f"질문: {enhanced_question}")

    return {"documents": documents, "question": enhanced_question}



# 답변 생성 노드
def generate(state):
    # print("==== [GENERATE] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]
    
        
    # 이미지 분석 결과가 없을 때는 원래 질문 그대로 사용
    if documents and isinstance(documents[0], str):
        enhanced_context = "\n\n".join(documents)
    else:
        enhanced_context = "\n\n".join([doc.page_content for doc in documents])


    # RAG 답변 생성
    generation = rag_chain.invoke({"context": enhanced_context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


# 문서 관련성 평가 노드
def grade_documents(state):
    # print("==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]

    # 각 문서에 대한 관련성 점수 계산
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            # {"question": question, "document": d.page_content}
            {"question": question, "document": d}
        )
        grade = score.binary_score
        if grade == "yes":
            # print("---GRADE: DOCUMENT RELEVANT---")
            # 관련성이 있는 문서 추가
            filtered_docs.append(d)
        else:
            # 관련성이 없는 문서는 건너뛰기
            # print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


# 질문 재작성 노드
def transform_query(state):
    # print("==== [TRANSFORM QUERY] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]

    # 질문 재작성
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


# 웹 검색 노드
def web_search(state):
    # print("==== [WEB SEARCH] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    image_result = state.get("image_result", "")

    if image_result:
        enhanced_question = f"{question} (병해충명 : {image_result})"
    else:
        enhanced_question = question

    # 웹 검색 수행
    web_results = web_search_tool.invoke({"query": enhanced_question})
    web_results_docs = [
        Document(
            page_content=web_result["content"],
            metadata={"source": web_result["url"]},
        )
        for web_result in web_results
    ]

    return {"documents": web_results_docs, "question": enhanced_question}

# 이미지 분석 노드
def image_analysis(state):
    """모델로 이미지 분석"""
    # print("==== [IMAGE ANALYSIS] ====")

    question = state["question"]
    image_data = state["image"]
    
    # 이미지 로드
    image = Image.open(image_data).convert('RGB')
    
    # 예측
    extended_inputs = extended_processor(image, return_tensors="pt")
    
    with torch.no_grad():
        extended_pred = torch.nn.functional.softmax(extended_model(**extended_inputs).logits[0], dim=-1)
    
    top_idx = extended_pred.argmax()
    
    if top_idx.item() in all_classes:
        class_name = all_classes[top_idx.item()]
    else:
        class_name = f"Unknown ({top_idx.item()})"
    # confidence = extended_pred[top_idx].item()
    # print(f"예측 결과: {class_name} (신뢰도: {confidence:.4f})")


    return {"question": question, "image_result": class_name, "documents": []}

# 질문 라우팅 노드
def route_question(state):
    # print("==== [ROUTE QUESTION] ====")
    # 질문 가져오기
    question = state["question"]
    image = state["image"]
    # 질문 라우팅

    if image and image != "" and image != None:
        return "image_analysis"


    source = question_router.invoke({"question": question})
    # 질문 라우팅 결과에 따른 노드 라우팅


    if source.datasource == "web_search":
        # print("==== [ROUTE QUESTION TO WEB SEARCH] ====")
        return "web_search"
    elif source.datasource == "vectorstore":
        # print("==== [ROUTE QUESTION TO VECTORSTORE] ====")
        return "vectorstore"


def image_route_question(state):
    # print("==== [IMAGE ROUTE QUESTION] ====")
     # 질문 가져오기
    question = state["question"]
    # 이미지 분석 결과 가져오기
    image_result = state["image_result"]

    # 직접 LLM 호출로 더 명확한 라우팅
    
    source = image_route_router.invoke({"question": question, "image_result": image_result})

    if source.datasource == "web_search":
        # print("==== [ROUTE QUESTION TO WEB SEARCH] ====")
        return "web_search"
    elif source.datasource == "vectorstore":
        # print("==== [ROUTE QUESTION TO VECTORSTORE] ====")
        return "vectorstore"



# 문서 관련성 평가 노드
def decide_to_generate(state):
    # print("==== [DECISION TO GENERATE] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # 모든 문서가 관련성 없는 경우 질문 재작성
        # print("==== [DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY] ====")
        return "transform_query"
    else:
        # 관련성 있는 문서가 있는 경우 답변 생성
        # print("==== [DECISION: GENERATE] ====")
        return "generate"


def hallucination_check(state):
    # print("==== [CHECK HALLUCINATIONS] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # 환각 평가
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Hallucination 여부 확인
    if grade == "yes":
        # print("==== [DECISION: GENERATION IS GROUNDED IN DOCUMENTS] ====")

        # 답변의 관련성(Relevance) 평가
        # print("==== [GRADE GENERATED ANSWER vs QUESTION] ====")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score

        # 관련성 평가 결과에 따른 처리
        if grade == "yes":
            # print("==== [DECISION: GENERATED ANSWER ADDRESSES QUESTION] ====")
            return "relevant"
        else:
            # print("==== [DECISION: GENERATED ANSWER DOES NOT ADDRESS QUESTION] ====")
            return "not relevant"
    else:
        # print("==== [DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY] ====")
        return "hallucination"


# 그래프 상태 초기화
workflow = StateGraph(GraphState)

# 노드 정의
workflow.add_node("web_search", web_search)  # 웹 검색
workflow.add_node("retrieve", retrieve)  # 문서 검색
workflow.add_node("grade_documents", grade_documents)  # 문서 평가
workflow.add_node("generate", generate)  # 답변 생성
workflow.add_node("transform_query", transform_query)  # 쿼리 변환
workflow.add_node("image_analysis", image_analysis) # 이미지 분석


# 그래프 빌드
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",  # 웹 검색으로 라우팅
        "vectorstore": "retrieve",  # 벡터스토어로 라우팅
        "image_analysis": "image_analysis",  # 이미지 분석으로 라우팅
    },
)

workflow.add_conditional_edges(
    "image_analysis",
    image_route_question,
    {
        "web_search": "web_search",  # 웹 검색으로 라우팅
        "vectorstore": "retrieve",  # 벡터스토어로 라우팅
    }
)

workflow.add_edge("web_search", "generate")  # 웹 검색 후 답변 생성
workflow.add_edge("retrieve", "grade_documents")  # 문서 검색 후 평가


workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",  # 쿼리 변환 필요
        "generate": "generate",  # 답변 생성 가능
    },
)
workflow.add_edge("transform_query", "retrieve")  # 쿼리 변환 후 문서 검색


workflow.add_conditional_edges(
    "generate",
    hallucination_check,
    {
        "hallucination": "generate",  # Hallucination 발생 시 재생성
        "relevant": END,  # 답변의 관련성 여부 통과
        "not relevant": "transform_query",  # 답변의 관련성 여부 통과 실패 시 쿼리 변환
    },
)

# 그래프 컴파일
app = workflow.compile(checkpointer=MemorySaver())


# from langchain_teddynote.graphs import visualize_graph

# # 그래프 이미지 파일 저장
# visualize_graph(app)

# config 설정(재귀 최대 횟수, thread_id)
config = RunnableConfig(recursion_limit=7, configurable={"thread_id": uuid.uuid4()})


# 질문 입력
inputs = {
    "question": "작물에 병이 생겼는데 처방법을 알려줘",
    "image": "strawberry_data/Strawberry Disease Data/test/images/gray_mold399.jpg",
}

# 그래프 실행
stream_graph(app, inputs, config, ["agent", "rewrite", "generate"])

