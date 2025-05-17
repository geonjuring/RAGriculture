import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_teddynote.document_loaders import HWPLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from typing import List, Dict
import warnings

warnings.filterwarnings("ignore")


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# 1. HWP, PDF 문서 로딩 및 metadata 삽입
def load_docs_with_metadata(path: str, crop: str) -> List:
    # 파일 형식 분류
    if path.lower().endswith(".pdf"):
        loader = PDFPlumberLoader(path)
    elif path.lower().endswith(".hwp"):
        loader = HWPLoader(path)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

    docs = loader.load()
    for i, doc in enumerate(docs):
         doc.metadata.update({
            "작물": crop,
            "출처": os.path.basename(path),
            "페이지": i
        })

    return docs

# 2. 사용자 입력에서 필터 조건 추출
def extract_filter_from_query(query: str) -> Dict:
    crop_match = re.search(r"(딸기|토마토|망고)", query)

    filter_dict = {}
    if crop_match:
        filter_dict["작물"] = crop_match.group(1)
    return filter_dict


# 3. 웹 검색 툴
def web_search_with_notice(query: str) -> str:
    results = TavilySearchResults().invoke(query)
    return str(results) + "\n\n🔎 웹 검색 결과를 기반으로 제공된 정보입니다."

web_search_tool = Tool(
    name="web_search_tool",
    func=web_search_with_notice,
    description=(
        "딸기/토마토/망고에 대한 재배 정보가 문서에 없을 경우, 웹에서 검색을 수행합니다. "
        "웹에서 가져온 최신 정보가 필요할 때 사용하세요. "
        "사용 시 '🔎 웹 검색 결과를 기반으로 제공된 정보입니다.' 문구가 반드시 출력되어야 합니다."
    )
)

# 4. Chroma 벡터 DB 처리
def get_vectorstore(paths: List[str], index_path: str, crop: str):
    # 인덱스가 존재하면 로드, 없으면 None
    vs = Chroma(persist_directory=index_path,embedding_function=embeddings) if os.path.exists(index_path) else None

    all_docs = []
    for p in paths:
        all_docs.extend(load_docs_with_metadata(p, crop))

    split_docs = text_splitter.split_documents(all_docs)
    
    # 최초 생성시
    if vs is None:
        vs = Chroma.from_documents(split_docs, embeddings, persist_directory=index_path)
    
    # 기존 인덱스에 추가
    else:
        vs.add_documents(split_docs)
        vs.persist()

    return vs


# 5. 필터 기반 retriever tool 래핑
def make_filtered_tool(name, vectorstore, crop_name):
    retriever = vectorstore.as_retriever()

    def _filtered_search(query: str) -> str:
        filters = extract_filter_from_query(query)
        
        if filters:
            return retriever.invoke(query, filter=filters)

        return retriever.invoke(query)

    return Tool(
        name=name,
        func=_filtered_search,
        description=(
            f"{crop_name}에 대한 재배 정보를 제공합니다"
        )
    )

strawberry_paths = [
    r"Data\딸기(촉성재배) 농작업일정.hwp",
    r"Data\과즙이풍부한왕딸기킹스베리재배매뉴얼.pdf",
]
tomato_paths = [
    r"Data\토마토,방울토마토 농작업일정.hwp", 
    r"Data\29 토마토_저화질_단면.pdf",
    r"Data\과채류(토마토).pdf"
]
mango_paths = [
    r"Data\아열대과수(망고).pdf",
    r"Data\2023농업기술길잡이_망고_단면.pdf",
]

strawberry_index_path = "chroma_index/strawberry"
tomato_index_path = "chroma_index/tomato"
mango_index_path = "chroma_index/mango"


strawberry_vectorstore = get_vectorstore(strawberry_paths, strawberry_index_path, "딸기")
tomato_vectorstore = get_vectorstore(tomato_paths, tomato_index_path, "토마토")
mango_vectorstore = get_vectorstore(mango_paths, mango_index_path, "망고")


strawberry_retriever_tool = make_filtered_tool("strawberry", strawberry_vectorstore, "딸기")
tomato_retriever_tool = make_filtered_tool("tomato", tomato_vectorstore, "토마토")
mango_retriver_tool = make_filtered_tool("mango", mango_vectorstore, "망고")


tools = [
    strawberry_retriever_tool,
    tomato_retriever_tool,
    mango_retriver_tool,
    web_search_tool,
]

# 프롬프트 정의
prompt = ChatPromptTemplate.from_messages([
    ("system",
        """
당신은 작물 재배 전문가입니다. 사용자의 질문 의도를 파악해 아래 규칙에 따라 작물 재배에 필요한 정보를 정확하고 실용적으로 출력하세요.

질문 범주:

출력 형식:
작물 재배법이 궁금한 경우:
🌱 재배 시기
🌍 재배 환경 조건 (기후, 토양 등)
📏 재식 간격 및 정식 방법
💧 관수 방법
🧪 시비(비료) 방법
✂️ 생육 관리(가지치기, 유인 등)
🌾 수확 시기 및 방법


도구 사용 지침:
- 관련 정보는 crop_retriever_tool 또는 지역별 retriever_tool을 통해 검색하세요.
- 정보가 부족한 경우 web_search_tool을 사용하여 보완하세요.
- 웹 검색 툴을 사용했다면 결과 하단에 "🔎 웹 검색 결과를 기반으로 제공된 정보입니다."를 추가하세요.
- 초보 농업인도 이해할 수 있도록 간결하고 쉬운 언어로 설명하세요.

참고:

상세 사용법은 농촌진흥청 자료를 참고하세요.
"""
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Agent 실행기 구성
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

response = agent_executor.invoke({
    "input": "토마토 재배 대한 정보를 알려줘"
})

# Markup 형식 출력
# print(response["output"].strip()) 

# Markup 형식의 #, *을 제외하고 출력
response_text = response["output"].strip()
cleaned_response = re.sub(r'([*#])', '', response_text)

print(cleaned_response)