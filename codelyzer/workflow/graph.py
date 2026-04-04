from typing import TypedDict, Annotated, Sequence, Dict, Any
from abc import ABC, abstractmethod
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import structlog
from pathlib import Path

from .state import WorkflowState
from codelyzer.config import settings

logger = structlog.get_logger(__name__)


class WorkflowStep(ABC):
    """Base class for workflow execution steps."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique step name."""
        pass

    @abstractmethod
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute step modifying state. Return updated state."""
        pass

    def next_steps(self, state: WorkflowState) -> Sequence[str]:
        """List steps that should run after this one completes."""
        return []

    def should_run(self, state: WorkflowState) -> bool:
        """Condition to check if this step should execute."""
        return True


class LangGraphWorkflow:
    """LangGraph based workflow orchestration engine."""

    def __init__(self, checkpoint: bool = True):
        self.builder = StateGraph(WorkflowState)
        self.steps: Dict[str, WorkflowStep] = {}

        # Setup checkpoint persistence
        if checkpoint:
            checkpoint_dir = Path.home() / ".cache" / "codelyzer" / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.memory = SqliteSaver.from_conn_string(str(checkpoint_dir / "workflow.db"))
        else:
            self.memory = None

        self.graph = None

    def add_step(self, step: WorkflowStep):
        """Add step to workflow graph."""
        self.steps[step.name] = step

        def node_func(state: WorkflowState) -> dict:
            logger.info("Executing workflow step", step=step.name)
            try:
                updated_state = step.execute(state)
                updated_state.mark_step_complete(step.name, success=True)
                logger.debug("Step completed successfully", step=step.name)
                return updated_state.__dict__
            except Exception as e:
                state.add_error(f"Step {step.name} failed: {str(e)}")
                state.mark_step_complete(step.name, success=False)
                logger.exception("Step execution failed", step=step.name, error=str(e))
                return state.__dict__

        self.builder.add_node(step.name, node_func)
        logger.debug("Added workflow step", step=step.name)

    def add_edge(self, from_step: str, to_step: str):
        """Add directed edge between steps."""
        if from_step not in self.steps or to_step not in self.steps:
            raise ValueError(f"Unknown step in edge: {from_step} -> {to_step}")

        self.builder.add_edge(from_step, to_step)

    def add_conditional_edge(self, from_step: str, condition):
        """Add conditional branching edge."""
        self.builder.add_conditional_edges(from_step, condition)

    def set_entry_point(self, step_name: str):
        """Set workflow entry point."""
        if step_name not in self.steps:
            raise ValueError(f"Unknown entry point step: {step_name}")
        self.builder.set_entry_point(step_name)

    def set_finish_step(self, step_name: str):
        """Set step that leads to workflow completion."""
        if step_name not in self.steps:
            raise ValueError(f"Unknown finish step: {step_name}")
        self.builder.add_edge(step_name, END)

    def compile(self):
        """Compile workflow graph for execution."""
        self.graph = self.builder.compile(checkpointer=self.memory)
        logger.info("Workflow compiled successfully", steps=len(self.steps))

    def execute(self, initial_state: WorkflowState, thread_id: str = None) -> WorkflowState:
        """Execute workflow with given initial state."""
        if not self.graph:
            self.compile()

        logger.info("Starting workflow execution")

        config = {"configurable": {"thread_id": thread_id}} if thread_id and self.memory else None

        final_state = self.graph.invoke(initial_state.__dict__, config=config)

        # Reconstruct WorkflowState object from final state
        state = WorkflowState(**final_state)

        logger.info("Workflow completed",
                    succeeded=sum(state.step_results.values()),
                    failed=len(state.step_results) - sum(state.step_results.values()),
                    errors=len(state.errors))

        return state

    def draw_graph(self, output_path: Path):
        """Render workflow graph visualization."""
        if not self.graph:
            self.compile()
        output_path.write_bytes(self.graph.get_graph().draw_mermaid_png())
        logger.info("Workflow graph rendered", path=output_path)