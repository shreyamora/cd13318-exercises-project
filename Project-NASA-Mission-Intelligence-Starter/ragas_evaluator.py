from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    import asyncio

    try:
        # Create evaluator LLM with model gpt-3.5-turbo
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))

        # Create evaluator_embeddings with model text-embedding-3-small
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )

        # Define an instance for each metric to evaluate
        faithfulness = Faithfulness(llm=evaluator_llm)
        response_relevancy = ResponseRelevancy(
            llm=evaluator_llm, embeddings=evaluator_embeddings
        )

        # Build a single-turn sample from the provided data
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        # Evaluate the response using the metrics (async single-turn scoring)
        async def _score() -> Dict[str, float]:
            results: Dict[str, float] = {}
            results["faithfulness"] = await faithfulness.single_turn_ascore(sample)
            results["answer_relevancy"] = await response_relevancy.single_turn_ascore(sample)
            return results

        # Return the evaluation results
        return asyncio.run(_score())

    except Exception as e:
        return {"error": str(e)}
