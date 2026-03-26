from agent.workflow_engine import WorkflowEngine, WorkflowNode, NodeStatus


def test_basic_execution():
    nodes = [
        WorkflowNode(name="step1", executor=lambda s: {"result": "done"}),
        WorkflowNode(name="step2", executor=lambda s: {"result": s.get("step1", {}).get("result")}),
    ]
    engine = WorkflowEngine(nodes)
    results = engine.execute()

    assert results["step1"].status == NodeStatus.SUCCESS
    assert results["step2"].status == NodeStatus.SUCCESS


def test_required_node_failure_stops_workflow():
    def failing_executor(s):
        raise RuntimeError("intentional failure")

    nodes = [
        WorkflowNode(name="step1", executor=failing_executor, required=True, max_retries=0),
        WorkflowNode(name="step2", executor=lambda s: {"result": "ok"}),
    ]
    engine = WorkflowEngine(nodes)
    results = engine.execute()

    assert results["step1"].status == NodeStatus.FAILED
    assert "step2" not in results  # step2 不应该被执行


def test_optional_node_failure_continues():
    call_count = 0

    def failing_executor(s):
        raise RuntimeError("intentional failure")

    def counting_executor(s):
        nonlocal call_count
        call_count += 1
        return {"count": call_count}

    nodes = [
        WorkflowNode(name="step1", executor=failing_executor, required=False, max_retries=0),
        WorkflowNode(name="step2", executor=counting_executor, required=True),
    ]
    engine = WorkflowEngine(nodes)
    results = engine.execute()

    assert results["step1"].status == NodeStatus.FAILED
    assert results["step2"].status == NodeStatus.SUCCESS
    assert call_count == 1


def test_retry_mechanism():
    attempt = 0

    def flaky_executor(s):
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            raise RuntimeError(f"attempt {attempt} failed")
        return {"result": "finally worked"}

    nodes = [
        WorkflowNode(name="step1", executor=flaky_executor, max_retries=3),
    ]
    engine = WorkflowEngine(nodes)
    results = engine.execute()

    assert results["step1"].status == NodeStatus.SUCCESS
    assert results["step1"].retry_count == 1


def test_event_log():
    nodes = [
        WorkflowNode(name="step1", executor=lambda s: {"ok": True}),
    ]
    engine = WorkflowEngine(nodes)
    engine.execute()

    assert len(engine.events) >= 2  # started + succeeded
    event_types = [e.event_type for e in engine.events]
    assert "started" in event_types
    assert "succeeded" in event_types


def test_checkpoint():
    nodes = [
        WorkflowNode(name="step1", executor=lambda s: {"val": 1}),
        WorkflowNode(name="step2", executor=lambda s: {"val": 2}),
    ]
    engine = WorkflowEngine(nodes)
    engine.execute()

    checkpoint = engine.get_checkpoint()
    assert "step1" in checkpoint["completed"]
    assert "step2" in checkpoint["completed"]
