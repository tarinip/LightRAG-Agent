from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json_repair
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.utils import logger
from lightrag.prompt import PROMPTS

MAX_GRADER_RETRIES = 2


@dataclass
class GradeResult:
    """Result of grading retrieved content for a sub-task."""
    passed: bool
    relevance_score: float       # 0.0-1.0
    completeness_score: float    # 0.0-1.0
    sufficiency_score: float     # 0.0-1.0
    reasoning: str
    rewritten_query: Optional[str] = None
    missing_aspects: Optional[str] = None


class GraderAgent:
    """
    Evaluates retrieved content (entities, relationships, chunks) for
    relevance, completeness, and sufficiency before LLM synthesis.

    On failure, provides a rewritten query for the self-correction loop
    back to the retrieval step (Planner Rewriting).
    """

    def __init__(
        self,
        rag_instance: LightRAG,
        max_retries: int = MAX_GRADER_RETRIES,
    ):
        """
        Initialize the GraderAgent.

        Args:
            rag_instance: An initialized LightRAG instance (used for LLM calls).
            max_retries: Maximum number of query rewrites on grading failure.
        """
        self.rag = rag_instance
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    @staticmethod
    def summarize_retrieved_data(data: dict) -> dict:
        """Condense aquery_data result into a grader-friendly summary.

        Args:
            data: Raw result dict from ``rag.aquery_data()``.

        Returns:
            dict with keys: entities_summary, relationships_summary,
            chunks_summary, entity_count, relationship_count, chunk_count,
            processing_info.
        """
        result_data = data.get("data", {})
        metadata = data.get("metadata", {})

        entities = result_data.get("entities", [])
        relationships = result_data.get("relationships", [])
        chunks = result_data.get("chunks", [])

        entities_summary = "\n".join(
            f"- {e.get('entity_name', '?')} ({e.get('entity_type', '?')}): "
            f"{e.get('description', '')[:150]}"
            for e in entities[:30]
        ) or "No entities retrieved."

        relationships_summary = "\n".join(
            f"- {r.get('src_id', '?')} -> {r.get('tgt_id', '?')} "
            f"[{r.get('keywords', '')}]: {r.get('description', '')[:150]}"
            for r in relationships[:30]
        ) or "No relationships retrieved."

        chunks_summary = "\n".join(
            f"- Chunk {i+1}: {c.get('content', '')[:200]}..."
            for i, c in enumerate(chunks[:15])
        ) or "No document chunks retrieved."

        proc = metadata.get("processing_info", {})
        processing_info = (
            f"Total entities found: {proc.get('total_entities_found', 'N/A')}, "
            f"after truncation: {proc.get('entities_after_truncation', 'N/A')}. "
            f"Total relations found: {proc.get('total_relations_found', 'N/A')}, "
            f"after truncation: {proc.get('relations_after_truncation', 'N/A')}. "
            f"Final chunks: {proc.get('final_chunks_count', 'N/A')}."
        )

        return {
            "entities_summary": entities_summary,
            "relationships_summary": relationships_summary,
            "chunks_summary": chunks_summary,
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "chunk_count": len(chunks),
            "processing_info": processing_info,
        }

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    async def grade(
        self,
        subtask_query: str,
        parent_query: str,
        data: dict,
    ) -> GradeResult:
        """Grade retrieved content for relevance, completeness, and sufficiency.

        Args:
            subtask_query: The atomic sub-task query being evaluated.
            parent_query: The original user query (for context).
            data: Raw result dict from ``rag.aquery_data()``.

        Returns:
            GradeResult with scores and optional rewrite suggestion.
        """
        summary = self.summarize_retrieved_data(data)

        prompt = PROMPTS["grader_user_prompt"].format(
            parent_query=parent_query,
            subtask_query=subtask_query,
            entities_summary=summary["entities_summary"],
            relationships_summary=summary["relationships_summary"],
            chunks_summary=summary["chunks_summary"],
            entity_count=summary["entity_count"],
            relationship_count=summary["relationship_count"],
            chunk_count=summary["chunk_count"],
            processing_info=summary["processing_info"],
        )
        system_prompt = PROMPTS["grader_system_prompt"]

        response = await self.rag.llm_model_func(prompt, system_prompt=system_prompt)

        try:
            result = json_repair.loads(response)
            return GradeResult(
                passed=result.get("passed", False),
                relevance_score=float(result.get("relevance_score", 0.0)),
                completeness_score=float(result.get("completeness_score", 0.0)),
                sufficiency_score=float(result.get("sufficiency_score", 0.0)),
                reasoning=result.get("reasoning", "No reasoning"),
                rewritten_query=result.get("rewritten_query"),
                missing_aspects=result.get("missing_aspects"),
            )
        except Exception as e:
            logger.error(f"Grader parse error: {e}. Defaulting to pass.")
            return GradeResult(
                passed=True,
                relevance_score=0.5,
                completeness_score=0.5,
                sufficiency_score=0.5,
                reasoning=f"Grader parse error: {e}. Defaulting to pass.",
            )

    # ------------------------------------------------------------------
    # Answer-based grading (grades the synthesized answer, not retrieval)
    # ------------------------------------------------------------------

    async def grade_answer(
        self,
        query: str,
        answer: str,
    ) -> GradeResult:
        """Grade the synthesized answer for relevance, completeness, and sufficiency.

        Unlike ``grade()``, this evaluates the **final answer text** directly
        instead of re-retrieving data and grading retrieval quality.

        Args:
            query: The original user query.
            answer: The synthesized answer to evaluate.

        Returns:
            GradeResult with scores and optional rewrite suggestion.
        """
        prompt = PROMPTS["answer_grader_user_prompt"].format(
            query=query,
            answer=answer[:5000],
        )
        system_prompt = PROMPTS["answer_grader_system_prompt"]

        response = await self.rag.llm_model_func(prompt, system_prompt=system_prompt)

        try:
            result = json_repair.loads(response)
            return GradeResult(
                passed=result.get("passed", False),
                relevance_score=float(result.get("relevance_score", 0.0)),
                completeness_score=float(result.get("completeness_score", 0.0)),
                sufficiency_score=float(result.get("sufficiency_score", 0.0)),
                reasoning=result.get("reasoning", "No reasoning"),
                rewritten_query=result.get("rewritten_query"),
                missing_aspects=result.get("missing_aspects"),
            )
        except Exception as e:
            logger.error(f"Answer grader parse error: {e}. Defaulting to pass.")
            return GradeResult(
                passed=True,
                relevance_score=0.5,
                completeness_score=0.5,
                sufficiency_score=0.5,
                reasoning=f"Answer grader parse error: {e}. Defaulting to pass.",
            )

    # ------------------------------------------------------------------
    # Synthesis from raw data
    # ------------------------------------------------------------------

    async def synthesize_from_data(
        self,
        subtask_query: str,
        data: dict,
        user_prompt: str = "",
    ) -> str:
        """Synthesize an answer from raw retrieved data using the LLM.

        Called when the grader passes — converts structured retrieval data
        into a natural-language answer without going through ``rag.aquery()``.

        Args:
            subtask_query: The query to answer.
            data: Raw result dict from ``rag.aquery_data()``.
            user_prompt: Optional additional guidance for the LLM.

        Returns:
            Synthesized answer string.
        """
        result_data = data.get("data", {})

        entities = result_data.get("entities", [])
        relationships = result_data.get("relationships", [])
        chunks = result_data.get("chunks", [])

        entities_context = "\n".join(
            f"- {e.get('entity_name', '')} ({e.get('entity_type', '')}): {e.get('description', '')}"
            for e in entities
        ) or "No entities."

        relationships_context = "\n".join(
            f"- {r.get('src_id', '')} -> {r.get('tgt_id', '')} [{r.get('keywords', '')}]: {r.get('description', '')}"
            for r in relationships
        ) or "No relationships."

        chunks_context = "\n".join(
            f"- [Chunk {i+1}]: {c.get('content', '')}"
            for i, c in enumerate(chunks)
        ) or "No document chunks."

        prompt = PROMPTS["planner_subtask_synthesis"].format(
            entities_context=entities_context,
            relationships_context=relationships_context,
            chunks_context=chunks_context,
            subtask_query=subtask_query,
            user_prompt=user_prompt or "n/a",
        )

        return await self.rag.llm_model_func(prompt)
