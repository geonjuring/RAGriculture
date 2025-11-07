from rag.base import RetrievalChain
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from typing import List, Annotated
from typing import List, Dict, Annotated, Union
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dataclasses import dataclass
import os

'''
class PDFRetrievalChain(RetrievalChain):
    def __init__(self, source_uri: Annotated[str, "Source URI"]):
        self.source_uri = source_uri
        self.k = 10

    def load_documents(self, source_uris: List[str]):
        docs = []
        for source_uri in source_uris:
            loader = PDFPlumberLoader(source_uri)
            docs.extend(loader.load())

        return docs

    def create_text_splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
'''

'''class PDFRetrievalChain(RetrievalChain):
    def __init__(self, source_uri: Annotated[Union[str, List[str], List[Dict[str, str]]], "Source URI or list with optional crop info"]):
        self.source_uri = source_uri
        self.k = 10

    def load_documents(self, source_uris: Union[List[str], List[Dict[str, str]]]):
        docs = []
        for item in source_uris:
            if isinstance(item, str):
                loader = PDFPlumberLoader(item)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = item
                docs.extend(loaded_docs)
            elif isinstance(item, dict):
                path = item["path"]
                crop = item.get("crop", "unknown")
                loader = PDFPlumberLoader(path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = path
                    doc.metadata["crop"] = crop
                docs.extend(loaded_docs)
        return docs

    def create_text_splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        '''

@dataclass
class RetrievalChainResult:
    retriever: any
    chain: any


class PDFRetrievalChain:
    def __init__(
        self,
        source_uri: Annotated[Union[str, List[str], List[Dict[str, str]]], "PDF 파일 경로 or crop 포함된 리스트"],
        persist_dir: str = "db/crop_vector",
        k: int = 10,
        force_rebuild: bool = False,
        llm=None
    ):
        self.source_uri = source_uri
        self.persist_dir = persist_dir
        self.k = k
        self.force_rebuild = force_rebuild
        self.llm = llm or ChatOpenAI(model_name="gpt-4", temperature=0)

    def load_documents(self, source_uris):
        docs = []
        for item in source_uris:
            if isinstance(item, str):
                loader = PDFPlumberLoader(item)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = item
                docs.extend(loaded_docs)
            elif isinstance(item, dict):
                path = item["path"]
                crop = item.get("crop", "unknown")
                loader = PDFPlumberLoader(path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = path
                    doc.metadata["crop"] = crop
                docs.extend(loaded_docs)
        return docs

    def create_text_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,       # 500 → 300 (토큰 제한 해결)
            chunk_overlap=100      # 30 유지 (적절한 오버랩)
        )

    def create_chain(self):
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",  # small 모델 사용 (토큰 절약)
            chunk_size=200  # 500 → 200 (배치 크기 줄이기)
        )

        if os.path.exists(self.persist_dir) and not self.force_rebuild:
            print(f"🔁 기존 벡터 DB 로드 중: {self.persist_dir}")
            vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embedding_model
            )
        else:
            print("📄 PDF 로드 및 벡터화 수행 중...")
            docs = self.load_documents(self.source_uri)
            splitter = self.create_text_splitter()
            split_docs = splitter.split_documents(docs)

            print(f"💾 벡터 DB 저장 위치: {self.persist_dir}")
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embedding_model,
                persist_directory=self.persist_dir
            )
            vectorstore.persist()

        retriever = vectorstore.as_retriever(search_kwargs={"k": self.k})
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff"
        )

        return RetrievalChainResult(retriever=retriever, chain=chain)