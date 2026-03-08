"""
Agentic LightRAG Pipeline
==========================
Orchestrates the full agentic retrieval workflow:

    1. Complex Query (input)
    2. Classifier Agent -- determines if query needs decomposition
    3. Planner Agent -- decomposes complex queries into sub-tasks
    4. Sub-tasks (dependency-aware DAG)
    5-9. LightRAG retrieval per sub-task (entities, relationships, chunks)
    10. Final Synthesis -- merge all sub-task results
    11. Grader Agent -- evaluates the final answer
    12. Pass --> return answer
    13. Fail --> feedback to Planner --> re-plan --> loop back to step 4

Single entry point: ``await AgenticRAG(rag).run(query)``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Union, AsyncIterator
import asyncio
import json_repair

from lightrag.lightrag import LightRAG, QueryParam
from lightrag.grader_agent import GraderAgent, GradeResult, MAX_GRADER_RETRIES
from lightrag.utils import logger
from lightrag.prompt import PROMPTS


# =====================================================================
# Data classes
# =====================================================================

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


@dataclass
class TaskState:
    """Manages the state of the entire planning and execution process."""
    original_query: str
    tasks: Dict[str, SubTask] = field(default_factory=dict)
    global_context: str = ""
    start_time: float = field(default_factory=lambda: 0.0)
    end_time: float = field(default_factory=lambda: 0.0)
    final_grade: Optional[Dict[str, Any]] = None
    plan_attempts: int = 0
    grade_history: List[Dict[str, Any]] = field(default_factory=list)


# =====================================================================
# Constants
# =====================================================================

FALLBACK_MODES: list[Literal["hybrid", "local", "global", "naive"]] = [
    "hybrid", "local", "global", "naive",
]

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

MAX_PLAN_ATTEMPTS = 3

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


# =====================================================================
# AgenticRAG -- Full Pipeline Orchestrator
# =====================================================================

class AgenticRAG:
    """
    Full agentic pipeline over LightRAG.

    Workflow:
        classify -> plan -> execute (no per-subtask grading) -> synthesize
        -> grade final answer -> PASS: return | FAIL: re-plan with feedback -> loop

    Delegates final-answer grading to ``GraderAgent``.
    """

    def __init__(
        self,
        rag_instance: LightRAG,
        query_params: Optional[Dict[str, Any]] = None,
        fallback_params: Optional[Dict[str, Any]] = None,
        enable_grading: bool = True,
        max_plan_attempts: int = MAX_PLAN_ATTEMPTS,
    ):
        self.rag = rag_instance
        self.state: Optional[TaskState] = None
        self._query_params = query_params or PLANNER_QUERY_PARAMS
        self._fallback_params = fallback_params or FALLBACK_QUERY_PARAMS
        self._enable_grading = enable_grading
        self._max_plan_attempts = max_plan_attempts

        # Grader for final-answer evaluation
        self._grader = GraderAgent(
            rag_instance=rag_instance,
            max_retries=MAX_GRADER_RETRIES,
        )

    # =================================================================
    # Sub-task result summarization
    # =================================================================

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
            return result[:300] + ("..." if len(result) > 300 else "")

    # =================================================================
    # Helper: Build QueryParam
    # =================================================================

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

    # =================================================================
    # Step 2: Classifier Agent
    # =================================================================

    async def classify(self, query: str) -> bool:
        """Determines if a query is complex enough to require planning."""
        logger.info(f"Classifying query complexity: {query}")
        prompt = PROMPTS["query_classifier"].format(query=query)
        response = await self.rag.llm_model_func(prompt)

        try:
            result = json_repair.loads(response)
            is_complex = result.get("is_complex", False)
            reasoning = result.get("reasoning", "No reasoning provided")
            logger.info(
                f"Query complexity: {'Complex' if is_complex else 'Simple'} - {reasoning}"
            )
            return is_complex
        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            return True

    # =================================================================
    # Steps 3-4: Planner Agent (query decomposition into sub-task DAG)
    # =================================================================

    async def plan(
        self,
        query: str,
        feedback: Optional[str] = None,
    ) -> TaskState:
        """Decomposes the query into dependency-aware sub-tasks.

        If *feedback* is provided (from a failed grading round), it is
        appended to the planner prompt so the LLM can produce a better plan.
        """
        logger.info(f"Planning tasks for query: {query}")
        prompt = PROMPTS["planner_system_prompt"]
        user_prompt = PROMPTS["planner_user_prompt"].format(query=query)

        if feedback:
            user_prompt += (
                f"\n\n---IMPORTANT FEEDBACK FROM PREVIOUS ATTEMPT---\n"
                f"The previous plan produced an answer that was graded as INSUFFICIENT.\n"
                f"Grader feedback: {feedback}\n"
                f"Please create a BETTER plan that addresses the missing aspects.\n"
                f"Use different retrieval modes, more specific sub-queries, "
                f"or break the problem down differently."
            )

        response = await self.rag.llm_model_func(user_prompt, system_prompt=prompt)

        try:
            data = json_repair.loads(response)
            tasks_data = data.get("sub_tasks", [])
            state = TaskState(
                original_query=query,
                start_time=asyncio.get_event_loop().time(),
            )
            for task_dict in tasks_data:
                sub_task = SubTask(
                    id=task_dict["id"],
                    query=task_dict["query"],
                    mode=task_dict.get("mode", "hybrid"),
                    depends_on=task_dict.get("depends_on", []),
                )
                state.tasks[sub_task.id] = sub_task
            return state
        except Exception as e:
            logger.error(f"Error during planning: {e}")
            state = TaskState(
                original_query=query,
                start_time=asyncio.get_event_loop().time(),
            )
            state.tasks["task_1"] = SubTask(id="task_1", query=query, mode="hybrid")
            return state

    # =================================================================
    # Steps 5-9: Execute sub-tasks (straight retrieval, no grading)
    # =================================================================

    async def _execute_subtask(
        self,
        task: SubTask,
        augmented_query: str,
        user_prompt_hint: str,
        is_retry: bool = False,
    ) -> str:
        """Execute a single sub-task via rag.aquery (no per-subtask grading)."""
        param = self._build_query_param(
            mode=task.mode,
            is_fallback=is_retry,
            user_prompt=user_prompt_hint,
        )
        result = await self.rag.aquery(augmented_query, param=param)
        if isinstance(result, AsyncIterator):
            full_result = ""
            async for chunk in result:
                full_result += chunk
            result = full_result
        return result

    # =================================================================
    # Execution orchestrator (runs all sub-tasks in dependency order)
    # =================================================================

    async def execute(self, state: TaskState) -> tuple[str, list[str]]:
        """Run all sub-tasks and return (synthesized_answer, results_context)."""
        logger.info(f"Executing plan for: {state.original_query}")
        results_context = []

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
                    deps_results.append(
                        f"Q: {dep.query}\n"
                        f"A: {dep_answer}"
                    )
            context_str = "\n\n".join(deps_results)

            # Mode-specific query augmentation
            if context_str:
                if task.mode == "local":
                    augmented_query = (
                        f"Based on the following previous findings:\n{context_str}\n\n"
                        f"Please provide specific details and entities related to: {task.query}"
                    )
                elif task.mode == "global":
                    augmented_query = (
                        f"Given the context of these previous steps:\n{context_str}\n\n"
                        f"Provide a high-level summary or thematic connection for: {task.query}"
                    )
                elif task.mode == "naive":
                    augmented_query = (
                        f"Context:\n{context_str}\n\n"
                        f"Directly answer the question: {task.query}"
                    )
                else:  # hybrid or mix
                    augmented_query = (
                        f"Using the context below:\n{context_str}\n\nTask: {task.query}"
                    )
            else:
                augmented_query = task.query

            logger.info(f"Running sub-task {task.id}: {task.query} (mode: {task.mode})")

            user_prompt_hint = (
                f"This is a sub-task of a larger query: '{state.original_query}'. "
                f"Focus on thoroughly answering: '{task.query}'. "
                "Provide specific details, names, and descriptions from the source material."
            )

            # Execute sub-task
            if task.mode == "reasoning":
                prompt = (
                    "Based on the following context, please perform the task.\n"
                    "Provide a thorough and detailed response. If the context contains relevant\n"
                    "information, synthesize it carefully. If the context is insufficient for certain\n"
                    "aspects, clearly state what is known and what remains uncertain.\n\n"
                    f"Context:\n{context_str}\n\nTask: {task.query}\n"
                )
                result = await self.rag.llm_model_func(prompt)
            else:
                result = await self._execute_subtask(
                    task=task,
                    augmented_query=augmented_query,
                    user_prompt_hint=user_prompt_hint,
                    is_retry=(state.plan_attempts > 0),
                )

            # Handle AsyncIterator (safety net)
            if isinstance(result, AsyncIterator):
                full_result = ""
                async for chunk in result:
                    full_result += chunk
                result = full_result

            # Fallback: try other modes if result is empty
            if task.mode != "reasoning" and _is_empty_answer(str(result)):
                for fallback_mode in FALLBACK_MODES:
                    if fallback_mode == task.mode:
                        continue
                    logger.info(
                        f"Sub-task {task.id} got empty result, "
                        f"fallback with '{fallback_mode}'"
                    )
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

        # Final synthesis across all sub-tasks
        final_answer = await self._final_synthesis(state, results_context)

        return final_answer, results_context

    # =================================================================
    # Final Synthesis
    # =================================================================

    async def _final_synthesis(
        self,
        state: TaskState,
        results_context: list[str],
    ) -> str:
        """Synthesize a final answer from all sub-task results."""
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

        return await self.rag.llm_model_func(final_prompt)

    # =================================================================
    # Grade Final Answer
    # =================================================================

    async def _grade_final_answer(
        self,
        state: TaskState,
        answer: str,
    ) -> tuple[bool, Optional[GradeResult]]:
        """Grade the final synthesized answer. Returns (passed, grade_result).

        Evaluates the actual answer text against the query, rather than
        re-retrieving data and grading retrieval quality. This is more
        accurate because the agentic pipeline uses multi-step retrieval
        across different sub-tasks and modes — a single fresh retrieval
        cannot capture what the pipeline actually found.
        """
        grade = await self._grader.grade_answer(
            query=state.original_query,
            answer=answer,
        )

        logger.info(
            f"Final answer grade: passed={grade.passed}, "
            f"R={grade.relevance_score:.2f}, "
            f"C={grade.completeness_score:.2f}, "
            f"S={grade.sufficiency_score:.2f} "
            f"- {grade.reasoning}"
        )

        grade_record = {
            "attempt": state.plan_attempts + 1,
            "passed": grade.passed,
            "relevance": grade.relevance_score,
            "completeness": grade.completeness_score,
            "sufficiency": grade.sufficiency_score,
            "reasoning": grade.reasoning,
        }
        state.grade_history.append(grade_record)

        if grade.passed:
            state.final_grade = {**grade_record, "passed": True}
            return True, grade

        state.final_grade = {**grade_record, "passed": False}
        return False, grade

    # =================================================================
    # Topological Sort (DAG ordering)
    # =================================================================

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
            for dep_id in sorted(tasks[task_id].depends_on):
                if dep_id in tasks:
                    visit(dep_id)
            temp_visited.remove(task_id)
            visited.add(task_id)
            stack.append(task_id)

        for task_id in sorted(tasks.keys()):
            visit(task_id)
        return stack

    # =================================================================
    # Entry Point: Plan -> Execute -> Grade -> Re-plan loop
    # =================================================================

    async def run(self, query: str) -> Union[str, AsyncIterator[str]]:
        """Main entry point for the Agentic RAG pipeline.

        Flow:
            1. Classify (simple → direct query, complex → plan)
            2. Plan → Execute all sub-tasks → Synthesize final answer
            3. Grade the final answer
            4. PASS → return answer
            5. FAIL → feed grader feedback back to Planner → re-plan → loop (max 3)
        """
        is_complex = await self.classify(query)

        if not is_complex:
            logger.info("Query classified as simple. Bypassing planning.")
            param = self._build_query_param(mode="hybrid", is_fallback=False)
            result = await self.rag.aquery(query, param=param)
            if _is_empty_answer(str(result)):
                for fallback_mode in FALLBACK_MODES:
                    if fallback_mode == "hybrid":
                        continue
                    logger.info(
                        f"Simple query got empty result, retrying with '{fallback_mode}'"
                    )
                    param = self._build_query_param(mode=fallback_mode, is_fallback=True)
                    result = await self.rag.aquery(query, param=param)
                    if not _is_empty_answer(str(result)):
                        break
            return result

        # Complex query: Plan -> Execute -> Grade -> Re-plan loop
        logger.info("Query classified as complex. Running Agentic RAG pipeline.")

        feedback = None
        best_answer = None
        best_score = -1.0

        for attempt in range(self._max_plan_attempts):
            logger.info(
                f"Plan attempt {attempt + 1}/{self._max_plan_attempts}"
                + (f" (with grader feedback)" if feedback else "")
            )

            # Step 3-4: Plan (with feedback from previous failed grade)
            self.state = await self.plan(query, feedback=feedback)
            self.state.plan_attempts = attempt

            # Steps 5-9: Execute all sub-tasks (no per-subtask grading)
            answer, results_context = await self.execute(self.state)

            # Track the best answer across attempts
            if not _is_empty_answer(str(answer)):
                # Step 10-11: Grade the final answer
                if not self._enable_grading:
                    return answer

                passed, grade = await self._grade_final_answer(self.state, answer)

                # Track best answer by score
                if grade:
                    score = (
                        grade.relevance_score
                        + grade.completeness_score
                        + grade.sufficiency_score
                    )
                    if score > best_score:
                        best_score = score
                        best_answer = answer

                if passed:
                    logger.info(
                        f"Final answer PASSED grading on attempt {attempt + 1}."
                    )
                    return answer

                # FAIL: build feedback for re-planning
                if grade:
                    feedback = (
                        f"Relevance={grade.relevance_score:.2f}, "
                        f"Completeness={grade.completeness_score:.2f}, "
                        f"Sufficiency={grade.sufficiency_score:.2f}. "
                        f"Reasoning: {grade.reasoning}. "
                    )
                    if grade.missing_aspects:
                        feedback += f"Missing: {grade.missing_aspects}. "
                    if grade.rewritten_query:
                        feedback += (
                            f"Suggested rewritten query: {grade.rewritten_query}"
                        )

                logger.info(
                    f"Final answer FAILED grading on attempt {attempt + 1}. "
                    f"Re-planning with feedback."
                )
            else:
                logger.warning(
                    f"Attempt {attempt + 1} produced empty answer. Re-planning."
                )
                feedback = (
                    "The previous plan produced an empty or refusal answer. "
                    "Try different retrieval modes and more specific sub-queries."
                )

        # All plan attempts exhausted — return the best answer we got
        logger.warning(
            f"All {self._max_plan_attempts} plan attempts exhausted. "
            "Returning best answer."
        )
        if best_answer and not _is_empty_answer(str(best_answer)):
            return best_answer

        # Ultimate fallback: direct query
        logger.warning("No good answer from any attempt. Falling back to direct query.")
        param = self._build_query_param(mode="hybrid", is_fallback=True)
        return await self.rag.aquery(query, param=param)
