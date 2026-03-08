from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Union, AsyncIterator
import asyncio
import json
import json_repair
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.utils import logger
from lightrag.prompt import PROMPTS

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

@dataclass
class SubTask:
    """Represents a single step in a multi-step query plan."""
    id: str
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "reasoning"] = "hybrid"
    depends_on: List[str] = field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    result: Optional[str] = None
    result_summary: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    grader_attempts: int = 0
    grader_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class TaskState:
    """Manages the state of the entire planning and execution process."""
    original_query: str
    tasks: Dict[str, SubTask] = field(default_factory=dict)
    global_context: str = ""
    start_time: float = field(default_factory=lambda: 0.0)
    end_time: float = field(default_factory=lambda: 0.0)

FALLBACK_MODES: list[Literal["hybrid", "local", "global", "naive"]] = ["hybrid", "local", "global", "naive"]

# Deeper retrieval params for sub-task execution (1.5x LightRAG defaults)
PLANNER_QUERY_PARAMS = {
    "top_k": 60,
    "chunk_top_k": 30,
    "max_entity_tokens": 12000,
    "max_relation_tokens": 16000,
    "max_total_tokens": 50000,
}

# Even deeper params for fallback retries (2x defaults)
FALLBACK_QUERY_PARAMS = {
    "top_k": 80,
    "chunk_top_k": 40,
    "max_entity_tokens": 16000,
    "max_relation_tokens": 20000,
    "max_total_tokens": 64000,
}

MAX_GRADER_RETRIES = 2

EMPTY_ANSWER_PHRASES = [
    "i do not have enough information",
    "i don't have enough information",
    "not enough information",
    "cannot answer",
    "unable to answer",
    "no relevant information",
    "i cannot find",
    "no information available",
]


def _is_empty_answer(text: str) -> bool:
    """Check if a result is effectively empty or a refusal."""
    if not text or not text.strip():
        return True
    lower = text.lower().strip()
    for phrase in EMPTY_ANSWER_PHRASES:
        if phrase in lower:
            return True
    return False


class PlannerAgent:
    """
    A specialized agent for decomposing complex queries into sub-tasks
    and executing them using LightRAG with dependency management.
    Includes a Grader Agent for retrieval quality assessment and
    self-correction via query rewriting.
    """
    def __init__(
        self,
        rag_instance: LightRAG,
        query_params: Optional[Dict[str, Any]] = None,
        fallback_params: Optional[Dict[str, Any]] = None,
        enable_grading: bool = True,
        max_grader_retries: int = MAX_GRADER_RETRIES,
    ):
        """
        Initialize the PlannerAgent.

        Args:
            rag_instance: An initialized LightRAG instance.
            query_params: Optional QueryParam overrides for sub-tasks.
            fallback_params: Optional QueryParam overrides for fallback retries.
            enable_grading: Whether to enable the Grader Agent for retrieval quality checks.
            max_grader_retries: Maximum number of query rewrites per sub-task on grading failure.
        """
        self.rag = rag_instance
        self.state: Optional[TaskState] = None
        self._query_params = query_params or PLANNER_QUERY_PARAMS
        self._fallback_params = fallback_params or FALLBACK_QUERY_PARAMS
        self._enable_grading = enable_grading
        self._max_grader_retries = max_grader_retries

    def _build_query_param(
        self,
        mode: str,
        is_fallback: bool = False,
        user_prompt: Optional[str] = None,
    ) -> QueryParam:
        """Build a QueryParam with deeper retrieval settings."""
        params = self._fallback_params if is_fallback else self._query_params
        return QueryParam(
            mode=mode,
            top_k=params.get("top_k", 60),
            chunk_top_k=params.get("chunk_top_k", 30),
            max_entity_tokens=params.get("max_entity_tokens", 12000),
            max_relation_tokens=params.get("max_relation_tokens", 16000),
            max_total_tokens=params.get("max_total_tokens", 50000),
            user_prompt=user_prompt,
        )

    # ------------------------------------------------------------------
    # Sub-task result summarization
    # ------------------------------------------------------------------

    async def _summarize_result(self, task_query: str, result: str) -> str:
        """Produce a concise 2-3 sentence summary of a sub-task result."""
        if not result or _is_empty_answer(result):
            return "No relevant information found."
        prompt = (
            "Summarize the following answer in 2-3 concise sentences. "
            "Keep specific names, numbers, and key facts. "
            "Do NOT add information that is not in the answer.\n\n"
            f"Question: {task_query}\n\n"
            f"Answer:\n{result[:3000]}\n\n"
            "Summary:"
        )
        try:
            summary = await self.rag.llm_model_func(prompt)
            return str(summary).strip()
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: truncate the result
            return result[:300] + ("..." if len(result) > 300 else "")

    # ------------------------------------------------------------------
    # Grader Agent methods
    # ------------------------------------------------------------------

    def _summarize_retrieved_data(self, data: dict) -> dict:
        """Condense aquery_data result into a grader-friendly summary."""
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

    async def _grade_retrieved_content(
        self,
        subtask_query: str,
        parent_query: str,
        data: dict,
    ) -> GradeResult:
        """Grade the retrieved content for relevance, completeness, and sufficiency."""
        summary = self._summarize_retrieved_data(data)

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

    async def _synthesize_from_data(
        self,
        subtask_query: str,
        data: dict,
        user_prompt: str = "",
    ) -> str:
        """Synthesize an answer from raw retrieved data using the LLM."""
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

    async def _execute_subtask_with_grading(
        self,
        task: SubTask,
        augmented_query: str,
        state: TaskState,
        user_prompt_hint: str,
    ) -> str:
        """Execute a sub-task with grading and self-correction loop."""
        # If grading is disabled, use the original direct path
        if not self._enable_grading:
            param = self._build_query_param(
                mode=task.mode, is_fallback=False, user_prompt=user_prompt_hint,
            )
            result = await self.rag.aquery(augmented_query, param=param)
            if isinstance(result, AsyncIterator):
                full_result = ""
                async for chunk in result:
                    full_result += chunk
                result = full_result
            return result

        current_query = augmented_query

        for attempt in range(1 + self._max_grader_retries):
            task.grader_attempts = attempt + 1

            # Phase A: Retrieve raw data
            param = self._build_query_param(
                mode=task.mode,
                is_fallback=(attempt > 0),
                user_prompt=user_prompt_hint,
            )
            data = await self.rag.aquery_data(current_query, param=param)

            # Check for total retrieval failure
            if data.get("status") != "success":
                logger.warning(f"Sub-task {task.id} retrieval failed on attempt {attempt + 1}")
                task.grader_history.append({
                    "attempt": attempt + 1,
                    "query": current_query,
                    "grade": None,
                    "data_summary": "Retrieval returned failure status",
                })
                if attempt < self._max_grader_retries:
                    current_query = task.query
                    continue
                else:
                    break

            # Phase B: Grade the retrieved content
            grade = await self._grade_retrieved_content(
                subtask_query=task.query,
                parent_query=state.original_query,
                data=data,
            )

            summary = self._summarize_retrieved_data(data)
            task.grader_history.append({
                "attempt": attempt + 1,
                "query": current_query,
                "grade": {
                    "passed": grade.passed,
                    "relevance": grade.relevance_score,
                    "completeness": grade.completeness_score,
                    "sufficiency": grade.sufficiency_score,
                    "reasoning": grade.reasoning,
                },
                "data_summary": (
                    f"{summary['entity_count']} entities, "
                    f"{summary['relationship_count']} relationships, "
                    f"{summary['chunk_count']} chunks"
                ),
            })

            logger.info(
                f"Sub-task {task.id} grade (attempt {attempt + 1}): "
                f"passed={grade.passed}, "
                f"R={grade.relevance_score:.2f}, "
                f"C={grade.completeness_score:.2f}, "
                f"S={grade.sufficiency_score:.2f} "
                f"- {grade.reasoning}"
            )

            if grade.passed:
                # Phase C (Pass): Synthesize answer from retrieved data
                result = await self._synthesize_from_data(
                    subtask_query=augmented_query,
                    data=data,
                    user_prompt=user_prompt_hint,
                )
                return result

            # Phase C (Fail): Rewrite query if retries remain
            if attempt < self._max_grader_retries:
                if grade.rewritten_query:
                    logger.info(
                        f"Sub-task {task.id} grading failed. "
                        f"Missing: {grade.missing_aspects}. "
                        f"Rewriting query: {grade.rewritten_query}"
                    )
                    current_query = grade.rewritten_query
                else:
                    logger.info(
                        f"Sub-task {task.id} grading failed without rewrite suggestion, "
                        "retrying with original query."
                    )
                    current_query = task.query

        # All grading attempts exhausted; fall back to direct rag.aquery()
        logger.warning(
            f"Sub-task {task.id}: grader exhausted {self._max_grader_retries + 1} attempts. "
            "Falling back to direct rag.aquery()."
        )
        param = self._build_query_param(
            mode=task.mode,
            is_fallback=True,
            user_prompt=user_prompt_hint,
        )
        result = await self.rag.aquery(augmented_query, param=param)
        if isinstance(result, AsyncIterator):
            full_result = ""
            async for chunk in result:
                full_result += chunk
            result = full_result
        return result

    # ------------------------------------------------------------------
    # Core pipeline methods
    # ------------------------------------------------------------------

    async def classify(self, query: str) -> bool:
        """Determines if a query is complex enough to require planning."""
        logger.info(f"Classifying query complexity: {query}")

        prompt = PROMPTS["query_classifier"].format(query=query)
        response = await self.rag.llm_model_func(prompt)

        try:
            result = json_repair.loads(response)
            is_complex = result.get("is_complex", False)
            reasoning = result.get("reasoning", "No reasoning provided")
            logger.info(f"Query complexity: {'Complex' if is_complex else 'Simple'} - {reasoning}")
            return is_complex
        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            return True

    async def plan(self, query: str) -> TaskState:
        """Decomposes the query into sub-tasks using the LLM."""
        logger.info(f"Planning tasks for query: {query}")

        prompt = PROMPTS["planner_system_prompt"]
        user_prompt = PROMPTS["planner_user_prompt"].format(query=query)

        response = await self.rag.llm_model_func(user_prompt, system_prompt=prompt)

        try:
            data = json_repair.loads(response)
            tasks_data = data.get("sub_tasks", [])

            state = TaskState(original_query=query, start_time=asyncio.get_event_loop().time())
            for task_dict in tasks_data:
                sub_task = SubTask(
                    id=task_dict["id"],
                    query=task_dict["query"],
                    mode=task_dict.get("mode", "hybrid"),
                    depends_on=task_dict.get("depends_on", [])
                )
                state.tasks[sub_task.id] = sub_task

            return state
        except Exception as e:
            logger.error(f"Error during planning: {e}")
            state = TaskState(original_query=query, start_time=asyncio.get_event_loop().time())
            state.tasks["task_1"] = SubTask(id="task_1", query=query, mode="hybrid")
            return state

    async def execute(self, state: TaskState) -> str:
        """Orchestrates the execution of sub-tasks with grading."""
        logger.info(f"Executing plan for: {state.original_query}")

        results_context = []

        # Sort tasks by dependencies (topological sort for DAG)
        sorted_task_ids = self._topological_sort(state.tasks)

        for task_id in sorted_task_ids:
            task = state.tasks[task_id]
            task.status = "running"

            # Inject previous results into context (use summaries for conciseness)
            deps_results = []
            for dep_id in task.depends_on:
                dep = state.tasks.get(dep_id)
                if dep and dep.result:
                    dep_answer = dep.result_summary or dep.result
                    deps_results.append(f"Q: {dep.query}\nA: {dep_answer}")

            context_str = "\n\n".join(deps_results)

            # Mode-specific query construction
            if context_str:
                if task.mode == "local":
                    augmented_query = f"Based on the following previous findings:\n{context_str}\n\nPlease provide specific details and entities related to: {task.query}"
                elif task.mode == "global":
                    augmented_query = f"Given the context of these previous steps:\n{context_str}\n\nProvide a high-level summary or thematic connection for: {task.query}"
                elif task.mode == "naive":
                    augmented_query = f"Context:\n{context_str}\n\nDirectly answer the question: {task.query}"
                else: # hybrid or mix
                    augmented_query = f"Using the context below:\n{context_str}\n\nTask: {task.query}"
            else:
                augmented_query = task.query

            logger.info(f"Running sub-task {task.id}: {task.query} (mode: {task.mode})")

            # Build user_prompt hint to guide retrieval
            user_prompt_hint = (
                f"This is a sub-task of a larger query: '{state.original_query}'. "
                f"Focus on thoroughly answering: '{task.query}'. "
                "Provide specific details, names, and descriptions from the source material."
            )

            # Execute sub-task using LightRAG or direct LLM for reasoning
            if task.mode == "reasoning":
                prompt = f"""Based on the following context, please perform the task.
Provide a thorough and detailed response. If the context contains relevant
information, synthesize it carefully. If the context is insufficient for certain
aspects, clearly state what is known and what remains uncertain.

Context:
{context_str}

Task: {task.query}
"""
                result = await self.rag.llm_model_func(prompt)
            else:
                # Use grader-aware execution (retrieve -> grade -> synthesize/rewrite)
                result = await self._execute_subtask_with_grading(
                    task=task,
                    augmented_query=augmented_query,
                    state=state,
                    user_prompt_hint=user_prompt_hint,
                )

            # Handle AsyncIterator if returned (safety net)
            if isinstance(result, AsyncIterator):
                full_result = ""
                async for chunk in result:
                    full_result += chunk
                result = full_result

            # Existing fallback: retry with different modes if result is empty
            if task.mode != "reasoning" and _is_empty_answer(str(result)):
                for fallback_mode in FALLBACK_MODES:
                    if fallback_mode == task.mode:
                        continue
                    logger.info(f"Sub-task {task.id} got empty result after grading, final fallback with '{fallback_mode}'")
                    param = self._build_query_param(
                        mode=fallback_mode,
                        is_fallback=True,
                        user_prompt=user_prompt_hint,
                    )
                    result = await self.rag.aquery(augmented_query, param=param)
                    if isinstance(result, AsyncIterator):
                        full_result = ""
                        async for chunk in result:
                            full_result += chunk
                        result = full_result
                    if not _is_empty_answer(str(result)):
                        task.mode = fallback_mode
                        break

            task.result = result
            task.result_summary = await self._summarize_result(task.query, str(result))
            task.status = "completed"
            results_context.append(f"Sub-task: {task.query}\nResult: {result}")

        # Final synthesis
        state.end_time = asyncio.get_event_loop().time()

        results_str = "\n\n---\n\n".join(results_context)
        final_prompt = f"""You are synthesizing a comprehensive answer from multiple retrieval sub-tasks.

Original Query: {state.original_query}

Sub-task Results:
{results_str}

---

Instructions:
1. Integrate information from ALL sub-tasks into a single coherent response.
2. Directly address every part of the original query.
3. Where sub-tasks found specific details (names, formulas, definitions), include them.
4. If any sub-task returned insufficient information, acknowledge the gap but still
   provide the best answer possible from the available evidence.
5. Use a structured format with clear paragraphs. Aim for thoroughness over brevity.
6. Do NOT say "I do not have enough information" unless truly no sub-task produced
   any relevant content at all.

Final Response:"""

        final_response = await self.rag.llm_model_func(final_prompt)
        return final_response

    def _topological_sort(self, tasks: Dict[str, SubTask]) -> List[str]:
        """Simple topological sort for task dependencies."""
        visited = set()
        temp_visited = set()
        stack = []

        def visit(task_id):
            if task_id in temp_visited:
                raise ValueError(f"Cycle detected in task dependencies: {task_id}")
            if task_id in visited:
                return

            temp_visited.add(task_id)
            # Sort depends_on to ensure stable output order
            for dep_id in sorted(tasks[task_id].depends_on):
                if dep_id in tasks:
                    visit(dep_id)

            temp_visited.remove(task_id)
            visited.add(task_id)
            stack.append(task_id)

        # Sort tasks by ID for stable starting point
        for task_id in sorted(tasks.keys()):
            visit(task_id)

        # The order in the stack is already from dependencies to dependents
        return stack

    async def run(self, query: str) -> Union[str, AsyncIterator[str]]:
        """Main entry point for the Planner Agent."""
        is_complex = await self.classify(query)

        if not is_complex:
            logger.info("Query classified as simple. Bypassing Planner Agent.")
            param = self._build_query_param(mode="hybrid", is_fallback=False)
            result = await self.rag.aquery(query, param=param)
            # If hybrid returns empty, try other modes with deeper params
            if _is_empty_answer(str(result)):
                for fallback_mode in FALLBACK_MODES:
                    if fallback_mode == "hybrid":
                        continue
                    logger.info(f"Simple query got empty result, retrying with '{fallback_mode}'")
                    param = self._build_query_param(mode=fallback_mode, is_fallback=True)
                    result = await self.rag.aquery(query, param=param)
                    if not _is_empty_answer(str(result)):
                        break
            return result

        logger.info("Query classified as complex. Proceeding with Planner Agent.")
        self.state = await self.plan(query)
        result = await self.execute(self.state)
        return result
