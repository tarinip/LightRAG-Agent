from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Union, AsyncIterator
import asyncio
import json_repair
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.utils import logger
from lightrag.prompt import PROMPTS

@dataclass
class SubTask:
    """Represents a single step in a multi-step query plan."""
    id: str
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "reasoning"] = "hybrid"
    depends_on: List[str] = field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    result: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None

@dataclass
class TaskState:
    """Manages the state of the entire planning and execution process."""
    original_query: str
    tasks: Dict[str, SubTask] = field(default_factory=dict)
    global_context: str = ""
    start_time: float = field(default_factory=lambda: 0.0)
    end_time: float = field(default_factory=lambda: 0.0)

class PlannerAgent:
    """
    A specialized agent for decomposing complex queries into sub-tasks
    and executing them using LightRAG with dependency management.
    """
    def __init__(self, rag_instance: LightRAG):
        """
        Initialize the PlannerAgent.

        Args:
            rag_instance (LightRAG): An initialized LightRAG instance.
        """
        self.rag = rag_instance
        self.state: Optional[TaskState] = None

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
            # Fallback to complex to be safe if decomposition fails
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
            # Fallback to a single sub-task if planning fails
            state = TaskState(original_query=query, start_time=asyncio.get_event_loop().time())
            state.tasks["task_1"] = SubTask(id="task_1", query=query, mode="hybrid")
            return state

    async def execute(self, state: TaskState) -> str:
        """Orchestrates the execution of sub-tasks."""
        logger.info(f"Executing plan for: {state.original_query}")

        results_context = []

        # Sort tasks by dependencies (topological sort for DAG)
        sorted_task_ids = self._topological_sort(state.tasks)

        for task_id in sorted_task_ids:
            task = state.tasks[task_id]
            task.status = "running"

            # Inject previous results into context if needed
            # We look at tasks this task depends on
            deps_results = []
            for dep_id in task.depends_on:
                if dep_id in state.tasks and state.tasks[dep_id].result:
                    deps_results.append(f"Q: {state.tasks[dep_id].query}\nA: {state.tasks[dep_id].result}")

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

            # Execute sub-task using LightRAG or direct LLM for reasoning
            if task.mode == "reasoning":
                prompt = f"""Based on the following context, please perform the task.

Context:
{context_str}

Task: {task.query}
"""
                result = await self.rag.llm_model_func(prompt)
            else:
                param = QueryParam(mode=task.mode)
                result = await self.rag.aquery(augmented_query, param=param)

            # Handle AsyncIterator if returned
            if isinstance(result, AsyncIterator):
                full_result = ""
                async for chunk in result:
                    full_result += chunk
                result = full_result

            task.result = result
            task.status = "completed"
            # Store both query and result for final synthesis
            results_context.append(f"Sub-task: {task.query}\nResult: {result}")

        # Final synthesis
        state.end_time = asyncio.get_event_loop().time()

        results_str = "\n".join(results_context)
        final_prompt = f"""Synthesize a final response for the original query based on the following sub-task results.
Original Query: {state.original_query}

Sub-task Results:
{results_str}

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
            return await self.rag.aquery(query)

        logger.info("Query classified as complex. Proceeding with Planner Agent.")
        self.state = await self.plan(query)
        result = await self.execute(self.state)
        return result
