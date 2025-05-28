import chess
import chess.pgn
import chess.engine
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def create_gameplay_agent(model, embeddings_model, vectorstore, finetuned_model, tokenizer, engine):
    """
    Create an agent to recommend a move by doing RAG from vectorstore
    
    Args:
        model: The language model to use for the agent
        embeddings_model: The embeddings model for vector operations
        vectorstore: The vector database containing chess moves
        engine: The Stockfish engine
        
    Returns:
        A recommendation agent configured with the provided resources
    """
    
    @tool
    def extract_fen(query: str) -> str:
        """Extract FEN notation from the user query"""
        import re
        # Look for a pattern that resembles FEN notation
        fen_pattern = r'[rnbqkpRNBQKP1-8]+/[rnbqkpRNBQKP1-8]+/[rnbqkpRNBQKP1-8]+/[rnbqkpRNBQKP1-8]+/[rnbqkpRNBQKP1-8]+/[rnbqkpRNBQKP1-8]+/[rnbqkpRNBQKP1-8]+/[rnbqkpRNBQKP1-8]+ [wb] [KQkq-]+ [a-h1-8-]+ \d+ \d+'
        match = re.search(fen_pattern, query)
        if match:
            return match.group(0)
        return "No FEN notation found in query"
    
    @tool
    def is_gradmaster_specific(query: str) -> str:
        """Get grandmaster name from the query"""
        import re
        pattern = r'carlsen|hikaru|gukesh|arjun|fabiano'
        match = re.search(pattern, str.lower(query))
        if match:
            return match.group(0)
        else:
            return "None"
            
    @tool
    def query_knowledge_base(name:str, fen: str) -> set:
        """Get relevant documents on a relevant vectorstore based on FEN from the query"""
        if name != "None":
            docs = vectorstore.similarity_search(fen, k=10)
        else:
            import re
            match = re.search(r"\s([wb])\s", fen)        
            if match:
                turn = match.group(1)
                color = "white" if turn == "w" else "black"
            
            results = vectorstore.similarity_search(fen, k=100)
            docs = [doc for doc in results if doc.metadata.get(color) == name][:10]
        return list(set([doc.metadata['move'] for doc in docs]))

    @tool
    def get_moves_for_white(move_history:str) -> str:
        """Get a move from a fine tuned model only when the turn is white"""
        
        messages = [{'content': 'You are a chess grandmaster.\n'
                                    'Please choose your next move.\n'
                                    'Use standard algebraic notation, e.g. "e4" or '
                                    '"Rdf8" or "R1a3".\n'
                                    'NEVER give a turn number.\n'
                                    'NEVER explain your choice.\n'
                                    'Here is a representation of the position:\n'
                                    'Your input is:\n'
                                    f'{move_history}',
                         'role': 'user'}]

        input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt = True,
                return_tensors = "pt",
            ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=100)

        generated_move = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_move
    
    @tool
    def evaluate_move(fen: str, move:str) -> int:
        """
        Calculate a centipawn score using Stockfish
        """
        board = chess.Board(fen)    
        board.push_san(move)
        score = engine.analyse(board, chess.engine.Limit(depth=15))["score"].relative.score()
        return score
    
    @tool
    def validate_move(fen:str, move:str) -> bool:
        """
        Validate whether the recommended move is legal or not on a current chess board state.
        """
        board = chess.Board(fen)
        try:
            board.push_san(move)
            return True
        except:
            return False
    
    tools = [extract_fen, is_gradmaster_specific, query_knowledge_base, get_moves_for_white, evaluate_move, validate_move, get_moves_for_white]
    
    prompt = """
    You are an intelligent agent that recommend chess moves to users.
    Follow these steps to answer user queries:
    1. Use the `extract_fen` tool to extract FEN from the user query.
    2. Use the `is_gradmaster_specific` tool to decide whether the user query is gradmaster specific or not.
    3. Use the `query_knowledge_base` tool to retrieve relevant documents from proper vectorstores.
    4. Use the `get_moves_for_white` tool to retrieve a move from fine-tuned model when the turn is white.
    5. Use the `evaluate_move` tool to calculate the centipawn score of the move.
    6. Use the 'validate_move' tool to make sure the recommended move is valid on the current board state.
    7. If you can't find enough information, state that explicitly.
    
    Use a step-by-step approach to complete each query.
    """
    gameplay_agent = create_react_agent(model, tools, prompt=prompt)
    
    return gameplay_agent