from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def create_coding_agent_graph(
    state_schema,
    recommend_steps_node,
    generate_code_node,
    execute_code_node,
    fix_code_node,
    should_generate_edge,
    should_retry_edge,
    generate_node_name="generate_code",
    execute_node_name="execute_code",
    fix_node_name="fix_code"
):
    """
    Factory function to create a state graph for a coding agent.
    """
    workflow = StateGraph(state_schema)

    workflow.add_node("recommend_steps", recommend_steps_node)
    workflow.add_node(generate_node_name, generate_code_node)
    workflow.add_node(execute_node_name, execute_code_node)
    workflow.add_node(fix_node_name, fix_code_node)

    workflow.set_entry_point("recommend_steps")

    workflow.add_conditional_edges(
        "recommend_steps",
        should_generate_edge,
        {
            "generate": generate_node_name,
            "wait": END
        }
    )

    workflow.add_edge(generate_node_name, execute_node_name)

    workflow.add_conditional_edges(
        execute_node_name,
        should_retry_edge,
        {
            "retry": fix_node_name,
            "end": END
        }
    )

    workflow.add_edge(fix_node_name, execute_node_name)

    return workflow.compile(
        checkpointer=MemorySaver(),
        interrupt_after=["recommend_steps"]
    )
