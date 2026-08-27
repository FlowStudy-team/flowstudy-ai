import asyncio
import unittest
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, ToolMessage

from app.agent import AgentLoop, AgentTrace, compact_messages


class FakeTool:
    name = "lookup"

    async def ainvoke(self, args):
        return f"found: {args['keyword']}"


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def ainvoke(self, _messages):
        return next(self.responses)


def collect(loop, messages, model, tools):
    async def run():
        return [event async for event in loop.run(messages, model, tools)]

    return asyncio.run(run())


class AgentLoopTests(unittest.TestCase):
    def test_tool_call_then_final_answer(self):
        responses = [
            SimpleNamespace(
                content="",
                tool_calls=[{"id": "call-1", "name": "lookup", "args": {"keyword": "arrays"}}],
            ),
            SimpleNamespace(content="Here is what I found.", tool_calls=[]),
        ]
        messages = [HumanMessage(content="Find arrays")]
        events = collect(AgentLoop(max_rounds=3), messages, FakeModel(responses), [FakeTool()])

        self.assertEqual([event.type for event in events], ["tool", "answer"])
        self.assertEqual(events[-1].content, "Here is what I found.")
        self.assertIsInstance(messages[-1], SimpleNamespace)
        self.assertIsInstance(messages[-2], ToolMessage)

    def test_unknown_tool_is_returned_as_observation(self):
        responses = [
            SimpleNamespace(content="", tool_calls=[{"id": "call-1", "name": "missing", "args": {}}]),
            SimpleNamespace(content="I could not access that tool.", tool_calls=[]),
        ]
        messages = [HumanMessage(content="Do it")]
        events = collect(AgentLoop(max_rounds=2), messages, FakeModel(responses), [])

        self.assertIn("Unknown tool: missing", messages[-2].content)
        self.assertEqual(events[-1].type, "answer")

    def test_max_rounds_stops_loop(self):
        response = SimpleNamespace(
            content="",
            tool_calls=[{"id": "call-1", "name": "lookup", "args": {"keyword": "x"}}],
        )
        messages = [HumanMessage(content="Keep going")]
        events = collect(AgentLoop(max_rounds=1), messages, FakeModel([response]), [FakeTool()])

        self.assertEqual(events[-1].type, "answer")
        self.assertIn("tool-call limit", events[-1].content)

    def test_context_compaction_does_not_start_with_tool_message(self):
        messages = [HumanMessage(content="old"), ToolMessage(content="result", tool_call_id="1")]
        compacted = compact_messages(messages, max_messages=1)
        self.assertEqual(compacted[0].type, "human")

    def test_trace_records_real_loop_observations(self):
        responses = [
            SimpleNamespace(content="", tool_calls=[{"id": "call-1", "name": "lookup", "args": {"keyword": "arrays"}}]),
            SimpleNamespace(content="done", tool_calls=[]),
        ]
        trace = AgentTrace()
        async def run():
            return [event async for event in AgentLoop().run([HumanMessage(content="Find")], FakeModel(responses), [FakeTool()], trace)]
        asyncio.run(run())
        self.assertEqual(trace.model_rounds, 2)
        self.assertEqual(trace.tool_calls[0].arguments, {"keyword": "arrays"})
        self.assertTrue(trace.tool_calls[0].success)


if __name__ == "__main__":
    unittest.main()
