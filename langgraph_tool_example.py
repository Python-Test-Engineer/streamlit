from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt.tools import tool
import operator
from pydantic import Field
from langgraph.utils import tool


# Define the state for our graph
class GraphState(TypedDict):
    query: str
    response: str
    math_result: float | None


# Define a simple tool using the @tool decorator
@tool
def calculate(
    operation: Annotated[
        Literal["add", "subtract", "multiply", "divide"],
        Field(description="The math operation to perform"),
    ],
    x: Annotated[float, Field(description="First number")],
    y: Annotated[float, Field(description="Second number")],
) -> float:
    """Perform a mathematical operation on two numbers."""

    operations = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
    }

    return operations[operation](x, y)


# Define our nodes
def generate_response(state: GraphState) -> GraphState:
    """Generate a response based on the query and math result."""
    query = state["query"]
    math_result = state["math_result"]

    if math_result is not None:
        response = f"The result of your calculation is {math_result}"
    else:
        response = "I didn't perform any calculations."

    return {"response": response}


# This function decides when to use our tool
def should_calculate(state: GraphState):
    """Determine if we should use the calculate tool."""
    query = state["query"].lower()

    # Simple logic to check if the query mentions math operations
    math_keywords = [
        "calculate",
        "add",
        "subtract",
        "multiply",
        "divide",
        "sum",
        "difference",
        "product",
    ]

    if any(keyword in query for keyword in math_keywords):
        return "calculate"
    else:
        return "generate_response"


# This function handles the tool call
def handle_calculation(state: GraphState) -> GraphState:
    """Parse the query and call the calculate tool."""
    query = state["query"].lower()

    # Very simplified parsing logic - in a real system, you'd use LLM or proper parsing
    if "add" in query or "sum" in query:
        # Extract numbers with very basic logic
        nums = [float(s) for s in query.split() if s.replace(".", "").isdigit()]
        if len(nums) >= 2:
            result = calculate(operation="add", x=nums[0], y=nums[1])
            return {"math_result": result}

    elif "subtract" in query or "difference" in query:
        nums = [float(s) for s in query.split() if s.replace(".", "").isdigit()]
        if len(nums) >= 2:
            result = calculate(operation="subtract", x=nums[0], y=nums[1])
            return {"math_result": result}

    elif "multiply" in query or "product" in query:
        nums = [float(s) for s in query.split() if s.replace(".", "").isdigit()]
        if len(nums) >= 2:
            result = calculate(operation="multiply", x=nums[0], y=nums[1])
            return {"math_result": result}

    elif "divide" in query:
        nums = [float(s) for s in query.split() if s.replace(".", "").isdigit()]
        if len(nums) >= 2 and nums[1] != 0:
            result = calculate(operation="divide", x=nums[0], y=nums[1])
            return {"math_result": result}

    # Default case if we can't parse the query
    return {"math_result": None}


# Build the graph
def build_graph():
    # Create a new graph
    graph = StateGraph(GraphState)

    # Add our nodes
    graph.add_node("should_calculate", should_calculate)
    graph.add_node("calculate", handle_calculation)
    graph.add_node("generate_response", generate_response)

    # Define the edges
    graph.add_edge("should_calculate", "calculate")
    graph.add_edge("should_calculate", "generate_response")
    graph.add_edge("calculate", "generate_response")
    graph.add_edge("generate_response", END)

    # Compile the graph
    return graph.compile()


# Example usage
if __name__ == "__main__":
    # Create an instance of our graph
    graph_instance = build_graph()

    # Process a query
    result = graph_instance.invoke(
        {"query": "Please add 5 and 7", "response": "", "math_result": None}
    )
    print(result["response"])  # Output: The result of your calculation is 12.0

    result = graph_instance.invoke(
        {"query": "What's the weather like?", "response": "", "math_result": None}
    )
    print(result["response"])  # Output: I didn't perform any calculations.
