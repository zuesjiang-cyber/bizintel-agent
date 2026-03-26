import time
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable, Dict, List, Any

from agent.schemas import NodeResult, NodeStatus, WorkflowEvent

logger = logging.getLogger(__name__)


class WorkflowNode:
    def __init__(
        self,
        name: str,
        executor: Callable[[dict], dict],
        required: bool = True,
        max_retries: int = 0,
        timeout_seconds: float = 60.0,
    ):
        self.name = name
        self.executor = executor
        self.required = required
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def execute(self, shared_state: dict) -> NodeResult:
        start_time = time.time()
        result = NodeResult(status=NodeStatus.RUNNING)

        for attempt in range(self.max_retries + 1):
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(self.executor, shared_state)
                    output = future.result(timeout=self.timeout_seconds)
                
                result.status = NodeStatus.SUCCESS
                result.data = output
                break
                
            except FuturesTimeoutError:
                result.error = f"Node timed out after {self.timeout_seconds:.1f}s"
                result.retry_count = attempt
                logger.error(f"Timeout in node {self.name} (attempt {attempt+1})")

                if attempt == self.max_retries:
                    result.status = NodeStatus.FAILED

            except Exception as e:
                logger.error(f"Error in node {self.name} (attempt {attempt+1}): {e}")
                result.error = str(e)
                result.retry_count = attempt
                
                if attempt == self.max_retries:
                    result.status = NodeStatus.FAILED

        result.duration_ms = int((time.time() - start_time) * 1000)
        return result


class WorkflowEngine:
    def __init__(self, nodes: List[WorkflowNode]):
        self.nodes = nodes
        self.events: List[WorkflowEvent] = []
        self._shared_state: Dict[str, Any] = {}
        self._results: Dict[str, NodeResult] = {}

    def execute(self, initial_state: dict = None) -> Dict[str, NodeResult]:
        """
        顺次执行所有节点，维护共享状态和事件日志
        返回每个节点的结果状态
        """
        self.events = []
        self._results = {}
        self._shared_state = initial_state or {}
        
        for node in self.nodes:
            self._log_event(node.name, "started")
            
            result = node.execute(self._shared_state)
            self._results[node.name] = result
            
            if result.status == NodeStatus.SUCCESS:
                self._log_event(node.name, "succeeded", f"duration: {result.duration_ms}ms")
                # 将结果写入共享状态，供后续节点使用
                if result.data:
                    self._shared_state[node.name] = result.data
                    
            elif result.status == NodeStatus.FAILED:
                self._log_event(node.name, "failed", f"error: {result.error}")
                
                if node.required:
                    logger.error(f"Required node {node.name} failed. Halting workflow.")
                    self._log_event("workflow", "halted", f"due to {node.name} failure")
                    break
                else:
                    logger.warning(f"Optional node {node.name} failed. Continuing.")
                    
        return self._results

    def get_checkpoint(self) -> dict:
        """返回当前状态快照，用于恢复（如果需要基于断点续传）"""
        return {
            "completed": [name for name, res in self._results.items() if res.status == NodeStatus.SUCCESS],
            "state": self._shared_state,
        }

    def _log_event(self, node_name: str, event_type: str, detail: str = None):
        event = WorkflowEvent(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            node_name=node_name,
            event_type=event_type,
            detail=detail,
        )
        self.events.append(event)
