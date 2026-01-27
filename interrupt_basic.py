from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


class GraphState(TypedDict):
    text: str


def start_node(state: GraphState) -> GraphState:
    print("▶ Start node")
    return state


def interrupt_node(state: GraphState) -> GraphState:
    print("⏸ About to interrupt")
    interrupt(
        {
            "reason": "Paused for inspection",
            "state": state,
        }
    )
    print("▶ Resumed after interrupt")
    return state


def end_node(state: GraphState) -> GraphState:
    print("▶ End node")
    return state


builder = StateGraph(GraphState)

builder.add_node("start", start_node)
builder.add_node("interrupt", interrupt_node)
builder.add_node("end", end_node)

builder.set_entry_point("start")
builder.add_edge("start", "interrupt")
builder.add_edge("interrupt", "end")
builder.add_edge("end", END)


state = {"text": "Hello LangGraph"}

# Execution PAUSES at interrupt
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Resume execution
config = {
    "configurable": {
        "thread_id": "debug-1"
    }
}
graph.invoke({"text": "Hello LangGraph"}, config=config)

graph.invoke(Command(resume={"text": "Hello LangGraph"}),
             config=config)
