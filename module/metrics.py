"""
RAG 품질 평가 메트릭스 모듈
"""
from typing import List, Dict, Optional
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from .config import debug_print


class RAGMetrics:
    """RAG 시스템의 품질을 정량적으로 평가하는 클래스
    
    LLM 기반 평가와 Embedding 기반 평가를 모두 지원합니다.
    LLM이 제공되면 LLM 기반 평가를 우선 사용하고, 그렇지 않으면 Embedding 기반 평가를 사용합니다.
    """
    
    def __init__(self, embedding_model, llm=None, grade_documents_grader=None, 
                 hallucination_grader=None, answer_grader=None):
        self.embedding_model = embedding_model
        self.llm = llm
        self.grade_documents_grader = grade_documents_grader
        self.hallucination_grader = hallucination_grader
        self.answer_grader = answer_grader
        self.use_llm_evaluation = llm is not None and all([
            grade_documents_grader, hallucination_grader, answer_grader
        ])
        
        if self.use_llm_evaluation:
            debug_print("✅ LLM 기반 평가 활성화")
        else:
            debug_print("⚠️ Embedding 기반 평가 사용 (LLM 또는 grader가 없음)")
    
    def calculate_retrieval_accuracy(self, question: str, retrieved_docs: List[Document]) -> float:
        """검색 정확도 계산 - 질문과 검색된 문서의 관련성"""
        if not retrieved_docs:
            return 0.0
        
        # LLM 기반 평가 사용
        if self.use_llm_evaluation and self.grade_documents_grader:
            try:
                doc_scores = []
                for doc in retrieved_docs:
                    evaluation = self.grade_documents_grader.invoke({
                        "question": question,
                        "document": doc.page_content
                    })
                    # binary_score: "yes" -> 1.0, "no" -> 0.0
                    score = 1.0 if evaluation.binary_score.lower() == "yes" else 0.0
                    doc_scores.append(score)
                
                return float(sum(doc_scores) / len(doc_scores)) if doc_scores else 0.0
            except Exception as e:
                debug_print(f"⚠️ LLM 기반 검색 정확도 평가 실패: {e}, Embedding 기반 평가로 전환")
        
        # Embedding 기반 평가 (Fallback)
        question_embedding = self.embedding_model.embed_query(question)
        total_similarity = 0
        
        for doc in retrieved_docs:
            doc_embedding = self.embedding_model.embed_query(doc.page_content)
            similarity = cosine_similarity([question_embedding], [doc_embedding])[0][0]
            total_similarity += similarity
        
        return float(total_similarity / len(retrieved_docs))
    
    def calculate_answer_relevance(self, question: str, answer: str) -> float:
        """답변 관련성 계산 - 질문과 답변의 관련성"""
        # LLM 기반 평가 사용
        if self.use_llm_evaluation and self.answer_grader:
            try:
                evaluation = self.answer_grader.invoke({
                    "question": question,
                    "generation": answer
                })
                # binary_score: "yes" -> 1.0, "no" -> 0.0
                return 1.0 if evaluation.binary_score.lower() == "yes" else 0.0
            except Exception as e:
                debug_print(f"⚠️ LLM 기반 답변 관련성 평가 실패: {e}, Embedding 기반 평가로 전환")
        
        # Embedding 기반 평가 (Fallback)
        question_embedding = self.embedding_model.embed_query(question)
        answer_embedding = self.embedding_model.embed_query(answer)
        
        similarity = cosine_similarity([question_embedding], [answer_embedding])[0][0]
        return float(similarity)
    
    def calculate_answer_correctness(self, answer: str, retrieved_docs: List[Document]) -> float:
        """답변 정확도 계산 - 답변이 검색된 문서에 근거하는지"""
        if not retrieved_docs:
            return 0.0
        
        # LLM 기반 평가 사용 (GradeDocuments 모델 사용)
        if self.use_llm_evaluation and self.grade_documents_grader:
            try:
                correctness_scores = []
                for doc in retrieved_docs:
                    # 답변이 문서에 근거하는지 평가
                    evaluation = self.grade_documents_grader.invoke({
                        "question": answer,  # 답변을 질문처럼 사용하여 문서와의 관련성 평가
                        "document": doc.page_content
                    })
                    score = 1.0 if evaluation.binary_score.lower() == "yes" else 0.0
                    correctness_scores.append(score)
                
                return float(sum(correctness_scores) / len(correctness_scores)) if correctness_scores else 0.0
            except Exception as e:
                debug_print(f"⚠️ LLM 기반 답변 정확도 평가 실패: {e}, Embedding 기반 평가로 전환")
        
        # Embedding 기반 평가 (Fallback)
        answer_embedding = self.embedding_model.embed_query(answer)
        total_similarity = 0
        
        for doc in retrieved_docs:
            doc_embedding = self.embedding_model.embed_query(doc.page_content)
            similarity = cosine_similarity([answer_embedding], [doc_embedding])[0][0]
            total_similarity += similarity
        
        return float(total_similarity / len(retrieved_docs))
    
    def calculate_hallucination_score(self, answer: str, retrieved_docs: List[Document]) -> float:
        """할루시네이션 점수 계산 - 답변이 문서에 근거하는지"""
        if not retrieved_docs:
            return 0.0
        
        # LLM 기반 평가 사용
        if self.use_llm_evaluation and self.hallucination_grader:
            try:
                # 모든 문서를 하나의 문자열로 결합
                documents_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                evaluation = self.hallucination_grader.invoke({
                    "documents": documents_text,
                    "generation": answer
                })
                # binary_score: "yes" -> 1.0 (사실 기반), "no" -> 0.0 (할루시네이션)
                return 1.0 if evaluation.binary_score.lower() == "yes" else 0.0
            except Exception as e:
                debug_print(f"⚠️ LLM 기반 할루시네이션 평가 실패: {e}, Embedding 기반 평가로 전환")
        
        # Embedding 기반 평가 (Fallback)
        # 답변의 각 문장이 문서에 근거하는지 확인
        answer_sentences = answer.split('.')
        total_score = 0
        
        for sentence in answer_sentences:
            if sentence.strip():
                sentence_embedding = self.embedding_model.embed_query(sentence.strip())
                max_similarity = 0
                
                for doc in retrieved_docs:
                    doc_embedding = self.embedding_model.embed_query(doc.page_content)
                    similarity = cosine_similarity([sentence_embedding], [doc_embedding])[0][0]
                    max_similarity = max(max_similarity, similarity)
                
                total_score += max_similarity
        
        return float(total_score / len(answer_sentences) if answer_sentences else 0.0)
    
    def evaluate(self, question: str, answer: str, documents: List[Document]) -> Dict[str, float]:
        """RAG 품질 종합 평가 (evaluate_rag_quality의 별칭)"""
        return self.evaluate_rag_quality(question, answer, documents)
    
    def evaluate_rag_quality(self, question: str, answer: str, retrieved_docs: List[Document]) -> Dict[str, float]:
        """RAG 품질 종합 평가
        
        LLM 기반 평가가 활성화되어 있으면 LLM 기반 점수를 사용하고,
        그렇지 않으면 Embedding 기반 점수를 사용합니다.
        
        각 메트릭 계산 함수가 자동으로 LLM 기반 평가를 우선 사용하므로,
        이 함수는 단순히 각 메트릭을 호출하여 종합 점수를 계산합니다.
        """
        # 각 메트릭 계산 (LLM 기반 평가 우선 사용, 실패 시 Embedding 기반 평가로 자동 전환)
        retrieval_accuracy = self.calculate_retrieval_accuracy(question, retrieved_docs)
        answer_relevance = self.calculate_answer_relevance(question, answer)
        answer_correctness = self.calculate_answer_correctness(answer, retrieved_docs)
        hallucination_score = self.calculate_hallucination_score(answer, retrieved_docs)
        
        # 전체 점수 계산 (가중 평균)
        # LLM 기반 평가: 0.0 또는 1.0 (binary)
        # Embedding 기반 평가: 0.0 ~ 1.0 (연속값)
        overall_score = (
            retrieval_accuracy * 0.3 +
            answer_relevance * 0.3 +
            answer_correctness * 0.2 +
            hallucination_score * 0.2
        )
        
        # 평가 방식 정보 추가
        evaluation_method = "LLM" if self.use_llm_evaluation else "Embedding"
        
        # numpy 타입을 Python 기본 타입으로 변환
        return {
            "retrieval_accuracy": float(retrieval_accuracy),
            "answer_relevance": float(answer_relevance),
            "answer_correctness": float(answer_correctness),
            "hallucination_score": float(hallucination_score),
            "overall_score": float(overall_score),
            "evaluation_method": evaluation_method  # 평가 방식 정보 추가
        }

