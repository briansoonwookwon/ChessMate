from typing import Literal, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage

def create_supervisor(model, recommendation_agent, knowledgebase_agent):
    members = ["recommender", "answerer"]

    options = members + ["FINISH"]
    system_prompt = f"""You are a supervisor agent responsible for routing user queries to specialized agents:{members}
    
    DECISION CRITERIA:
    - 'recommender': For queries seeking chess moves recommendation.
      Examples: "What move should I make?", "What can I do in this board?", "What would Hikaru do in this situation?"
    
    - 'answerer': For general queries asking about the chess
      Examples: "What are the common chess openings?", "Tell me about the mid game strategy", "When can I do a castling?"
    
    - '__end__': Only if the user explicitly wants to end the conversation or says goodbye.
      Examples: "That's all, thanks", "Goodbye", "We're done here"
    
    Your task is to analyze the latest user message and determine which specialized agent should handle it.
    Choose exactly one worker to handle this query.
    """

    class Router(TypedDict):
        """Worker to route to next."""
        next: Literal["recommender", "answerer"]
    
    
    class State(MessagesState):
        next: str

    def supervisor_node(state: State) -> Command[Literal["recommender", "answerer"]]:
        # Get existing messages
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        
        # Let the LLM decide which agent should handle this query
        response = model.with_structured_output(Router).invoke(messages)
        next_agent = response["next"]
        
        # Forward to the chosen agent
        return Command(goto=next_agent, update={"next": next_agent})



    def recommendation_node(state: State) -> Command[Literal["__end__"]]:
        result = recommendation_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="recommender")
                ]
            },
            goto="__end__",  # Go directly to end after this agent
        )
    
    def knowledgebase_node(state: State) -> Command[Literal["__end__"]]:
        result = knowledgebase_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="answerer")
                ]
            },
            goto="__end__",  # Go directly to end after this agent
        )



    builder = StateGraph(State)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("recommender", recommendation_node)
    builder.add_node("answerer", knowledgebase_node)
    builder.add_edge("recommender", "__end__")  # Direct edge from recommender to end
    builder.add_edge("answerer", "__end__")  # Direct edge from answerer to end
    
    supervisor = builder.compile()

    return supervisor






        