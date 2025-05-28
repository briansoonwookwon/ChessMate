# ---------- Bedrock Setup ----------
exec(open("bedrock_setup.py").read())

# ---------- Imports ----------
import streamlit as st
import chess
import chess.svg
import chess.engine
import re
from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import init_chat_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.puzzle_agent import create_puzzle_agent
from src.tutorial_agent import create_tutorial_agent
from src.gameplay_agent import create_gameplay_agent
from src.multi_agent_supervisor import create_supervisor

load_dotenv()

# ---------- Load necessary objects ----------
@st.cache_resource
def load_resources():
    # Embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    
    # Vectorstores
    all_moves = PineconeVectorStore.from_existing_index(
    index_name="chess-moves",
    embedding=embeddings_model
)

    # all_moves = FAISS.load_local("database/all_moves", embeddings_model, allow_dangerous_deserialization=True)
    kb = FAISS.load_local("database/knowledge_base", embeddings_model, allow_dangerous_deserialization=True)
    
    # LLM
    model = init_chat_model("us.anthropic.claude-3-5-haiku-20241022-v1:0",
                          model_provider="bedrock_converse",
                          region_name="us-east-1",
                          client=bedrock_client)
    
    # Stockfish
    recommendation_engine = chess.engine.SimpleEngine.popen_uci("Stockfish/src/stockfish")

    # Finetuned model
    model_id = "cdarkgtown/chess_llama_nogguf_2"
    finetuned_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Agents
    puzzle_agent = create_puzzle_agent()
    tutorial_agent = create_tutorial_agent(model, embeddings_model, kb)
    gameplay_agent = create_gameplay_agent(model, embeddings_model, all_moves, finetuned_model, tokenizer, recommendation_engine)
    supervisor = create_supervisor(model, gameplay_agent, tutorial_agent)
    
    return {
        "embeddings_model": embeddings_model,
        "all_moves": all_moves,
        "kb": kb,
        "model": model,
        "engine": recommendation_engine,
        "puzzle_agent": puzzle_agent,
        "tutorial_agent": tutorial_agent,
        "gameplay_agent": gameplay_agent,
        "supervisor": supervisor
    }

# ---------- Engine Management ----------
def get_engine():
    """
    Get or create a Stockfish engine instance.
    This function ensures we don't create multiple instances and properly manage the engine lifecycle.
    """
    # If engine doesn't exist in session state or was closed, create a new one
    if 'engine' not in st.session_state or st.session_state.engine is None:
        try:
            st.session_state.engine = chess.engine.SimpleEngine.popen_uci("Stockfish/src/stockfish")
        except Exception as e:
            st.error(f"Failed to start Stockfish engine: {str(e)}")
            return None
    
    return st.session_state.engine

def safely_quit_engine():
    """Safely quit the chess engine if it exists and is running"""
    if 'engine' in st.session_state and st.session_state.engine is not None:
        try:
            st.session_state.engine.quit()
        except chess.engine.EngineTerminatedError:
            # Engine already terminated, just ignore
            pass
        finally:
            st.session_state.engine = None

# ---------- Page Configuration ----------
def setup_page():
    st.set_page_config(
        page_title="ChessMate",
        page_icon="♟️",
        layout="wide",
        menu_items={
            'About': """
            ChessMate is your kind friend and a chess expert. You can ask any questions about chess and even play again it. 
            Also try to ask it to help you to beat Stockfish!
            """
        }
    )
    
    st.markdown(
        "<h1 style='text-align: center;'>♟️ ChessMate</h1>",
        unsafe_allow_html=True
    )

# ---------- State Management ----------
def initialize_state():
    if 'board' not in st.session_state:
        st.session_state.board = chess.Board()
        st.session_state.move_history = []
        st.session_state.selected_square = None
        st.session_state.board_key = 0
        st.session_state.move_error = None
        st.session_state.playing_stockfish = False
        st.session_state.playing_chessmate = False
        st.session_state.stockfish_color = chess.BLACK  # Default to Stockfish playing as black
        st.session_state.chessmate_color = chess.BLACK  # Default to ChessMate playing as black
        st.session_state.engine = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

# ---------- Chess Logic ----------
def make_move(move_text):
    """Process a move in either UCI or SAN format"""
    st.session_state.move_error = None
    
    try:
        # Try parsing as UCI (e.g., "e2e4")
        move = chess.Move.from_uci(move_text)
        if move in st.session_state.board.legal_moves:
            san_move = st.session_state.board.san(move)
            st.session_state.board.push(move)
            st.session_state.move_history.append(san_move)
            return True
    except ValueError:
        pass
    
    try:
        # Try parsing as SAN (e.g., "e4" or "Nf3")
        move = st.session_state.board.parse_san(move_text)
        if move in st.session_state.board.legal_moves:
            st.session_state.board.push(move)
            st.session_state.move_history.append(move_text)
            return True
    except ValueError:
        pass
    
    # If we got here, the move wasn't valid
    st.session_state.move_error = f"Invalid move: {move_text}"
    return False

def reset_game():
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.selected_square = None
    st.session_state.move_error = None
    st.session_state.playing_stockfish = False
    st.session_state.playing_chessmate = False
    
    # Close engine properly
    safely_quit_engine()

def load_puzzle():
    resources = load_resources()
    puzzle_agent = resources["puzzle_agent"]
    response = puzzle_agent.invoke({"messages": [HumanMessage(content="Get me a chess puzzle")]})
    
    puzzle_data = response["messages"][-1].content
    fen_pattern = r'[1-8rpnbqkRPNBQK/]+ [bw] [-kqKQ]+ [-\w]+ \d+ \d+'
    matches = re.findall(fen_pattern, puzzle_data)
    if matches:
        fen_string = matches[0]
        st.session_state.board = chess.Board(fen_string)
        st.session_state.move_history = []
        st.session_state.selected_square = None
        st.session_state.move_error = None
    else:
        st.session_state.move_error = "Failed to load puzzle. Please try again."

def play_against_stockfish(color):
    # Reset the game
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.selected_square = None
    st.session_state.move_error = None
    
    # Make sure any previous engine is properly closed
    safely_quit_engine()
    
    # Get a fresh engine instance
    st.session_state.engine = get_engine()
    if not st.session_state.engine:
        st.error("Could not initialize Stockfish engine")
        return
        
    st.session_state.playing_stockfish = True
    st.session_state.playing_chessmate = False
    st.session_state.stockfish_color = chess.BLACK if color == "white" else chess.WHITE
    
    # If Stockfish plays as white, make the first move
    if st.session_state.stockfish_color == chess.WHITE:
        make_stockfish_move()

def play_against_chessmate(color):
    # Reset the game
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.selected_square = None
    st.session_state.move_error = None
    
    # Close any existing engine
    safely_quit_engine()
    
    st.session_state.playing_stockfish = False
    st.session_state.playing_chessmate = True
    st.session_state.chessmate_color = chess.BLACK if color == "white" else chess.WHITE
    
    # If ChessMate plays as white, make the first move
    if st.session_state.chessmate_color == chess.WHITE:
        make_chessmate_move()

def make_stockfish_move():
    try:
        # Get or create engine if needed
        engine = get_engine()
        if not engine:
            st.session_state.move_error = "Stockfish engine not available"
            return False
            
        # Get Stockfish's move
        result = engine.play(
            st.session_state.board, 
            chess.engine.Limit(time=1.0)  # Give Stockfish 1 second to think
        )
        
        # Make the move
        move = result.move
        san_move = st.session_state.board.san(move)
        st.session_state.board.push(move)
        st.session_state.move_history.append(san_move)
        
        return True
    except chess.engine.EngineTerminatedError:
        # Engine crashed or was closed, try to restart it
        safely_quit_engine()
        st.session_state.engine = get_engine()
        st.session_state.move_error = "Engine restarted. Please try again."
        return False
    except Exception as e:
        st.session_state.move_error = f"Engine error: {str(e)}"
        return False

def make_chessmate_move():
    resources = load_resources()
    gameplay_agent = resources["gameplay_agent"]
    
    # Create a prompt asking for the best move
    prompt = f"What is the best move for {get_color_name(st.session_state.board.turn)} in this position? Current board state (FEN): {st.session_state.board.fen()}"
    
    # Get ChessMate's recommendation
    response = gameplay_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    response_content = response["messages"][-1].content
    
    # Extract the move from the response
    # Look for common move notation patterns
    move_patterns = [
        r'([a-h][1-8][a-h][1-8])',  # UCI format like e2e4
        r'([KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)'  # SAN format like Nf3, e4, or Qxd4+
    ]
    
    move_text = None
    for pattern in move_patterns:
        matches = re.findall(pattern, response_content)
        if matches:
            # Try each match until one works
            for match in matches:
                if make_move(match):
                    # Successfully made a move
                    return True
                    
    # If we couldn't extract a valid move, add message to chat explaining the issue
    if not move_text:
        error_message = "I'm having trouble deciding on a move. Please try again."
        st.session_state.messages.append({"role": "Assistant", "content": error_message})
        st.session_state.move_error = "ChessMate couldn't make a valid move"
        return False

def get_color_name(color):
    return "White" if color else "Black"

# ---------- UI Components ----------
def render_board():
    svg = chess.svg.board(
        board=st.session_state.board,
        size=400,
        lastmove=st.session_state.board.peek() if st.session_state.move_history else None,
        check=st.session_state.board.king(st.session_state.board.turn) if st.session_state.board.is_check() else None,
    )
    
    svg_html = f"""
    <div style="display: flex; justify-content: center;" key="{st.session_state.board_key}">
        {svg}
    </div>
    """
    st.markdown(svg_html, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.header("Game Controls")
        
        # Reset button
        if st.button("New Game"):
            reset_game()
            st.rerun()

        # Puzzle button
        if st.button("Puzzle Game"):
            load_puzzle()
            st.rerun()
        
        # Stockfish buttons
        st.header("Play against Stockfish")
        stock_col1, stock_col2 = st.columns(2)
        
        with stock_col1:
            if st.button("Play as White", key="stockfish_white"):
                play_against_stockfish("white")
                st.rerun()
        
        with stock_col2:
            if st.button("Play as Black", key="stockfish_black"):
                play_against_stockfish("black")
                st.rerun()
        
        # ChessMate buttons - new section
        st.header("Play against ChessMate")
        cm_col1, cm_col2 = st.columns(2)
        
        with cm_col1:
            if st.button("Play as White", key="chessmate_white"):
                play_against_chessmate("white")
                st.rerun()
        
        with cm_col2:
            if st.button("Play as Black", key="chessmate_black"):
                play_against_chessmate("black")
                st.rerun()
        
        # Display game status messages
        if st.session_state.playing_stockfish:
            st.success("Playing against Stockfish")
            engine_color = "Black" if st.session_state.stockfish_color == chess.BLACK else "White"
            st.write(f"Stockfish is playing as: {engine_color}")
        elif st.session_state.playing_chessmate:
            st.success("Playing against ChessMate")
            agent_color = "Black" if st.session_state.chessmate_color == chess.BLACK else "White"
            st.write(f"ChessMate is playing as: {agent_color}")
        
        # Display game status
        st.header("Game Status")
        if st.session_state.board.is_checkmate():
            st.error("Checkmate! Game over.")
        elif st.session_state.board.is_stalemate():
            st.warning("Stalemate! Game over.")
        elif st.session_state.board.is_check():
            st.warning("Check!")
        
        # Move history
        st.header("Move History")
        if st.session_state.move_history:
            move_df = {"White": [], "Black": []}
            for i, move in enumerate(st.session_state.move_history):
                if i % 2 == 0:
                    move_df["White"].append(move)
                else:
                    move_df["Black"].append(move)
            # Make sure Black has the same number of moves
            if len(move_df["White"]) > len(move_df["Black"]):
                move_df["Black"].append("")
            st.dataframe(move_df)
        else:
            st.text("No moves yet.")

def render_game_area():
    col1, spacer, col2 = st.columns([1, 0.1, 1])

    with col1:
        # Render the chess board
        render_board()
        
        # Move input
        st.write("Enter your move (UCI format like 'e2e4' or SAN format like 'e4', 'Nf3'):")
        move_col1, move_col2 = st.columns([3, 1])
        
        with move_col1:
            move_text = st.text_input("Move", key="move_input", label_visibility="collapsed")
        
        with move_col2:
            if st.button("Make Move"):
                if move_text:
                    if make_move(move_text):
                        st.session_state.board_key += 1
                        
                        # If playing against Stockfish and it's Stockfish's turn
                        if (st.session_state.playing_stockfish and 
                            st.session_state.board.turn == st.session_state.stockfish_color and
                            not st.session_state.board.is_game_over()):
                            make_stockfish_move()
                            st.session_state.board_key += 1
                            
                        # If playing against ChessMate and it's ChessMate's turn
                        elif (st.session_state.playing_chessmate and 
                            st.session_state.board.turn == st.session_state.chessmate_color and
                            not st.session_state.board.is_game_over()):
                            make_chessmate_move()
                            st.session_state.board_key += 1
                        
                        st.rerun()
        
        # Display error if any
        if st.session_state.move_error:
            st.error(st.session_state.move_error)
            
        # Show whose turn it is
        turn = "White" if st.session_state.board.turn else "Black"
        st.write(f"Current turn: {turn}")
        
        # Show additional details based on game mode
        if st.session_state.playing_stockfish:
            engine_color = "Black" if st.session_state.stockfish_color == chess.BLACK else "White"
            player_color = "White" if engine_color == "Black" else "Black"
            st.write(f"You are playing as: {player_color}")
            st.write(f"Stockfish is playing as: {engine_color}")
        elif st.session_state.playing_chessmate:
            agent_color = "Black" if st.session_state.chessmate_color == chess.BLACK else "White"
            player_color = "White" if agent_color == "Black" else "Black"
            st.write(f"You are playing as: {player_color}")
            st.write(f"ChessMate is playing as: {agent_color}")

    with col2.container(height=650):
        render_chat_interface()

def is_asking_for_move_recommendation(prompt):
    recommendation_keywords = [
        "recommend", "suggest", "best move", "good move", "what move", 
        "what's the best move", "what is the best move", "move recommendation",
        "what should I play", "how should I move", "next move", "what would"
    ]
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in recommendation_keywords)

def get_pgn_format_moves():
    game_temp = chess.pgn.Game.from_board(st.session_state.board)
    pgn_string = game_temp.accept(chess.pgn.StringExporter(headers=False))
    # remove the star symbol used for current move
    pgn_string = pgn_string.replace("*", "")
    return pgn_string

def render_chat_interface():
    resources = load_resources()
    supervisor = resources["supervisor"]
    
    # Create a container for chat messages with scrolling
    chat_container = st.container()
    
    # Create a fixed container at the bottom for the input
    input_container = st.container()
    
    # Handle new chat input (placing this before message display makes it appear at bottom)
    with input_container:
        if prompt := st.chat_input("Ask anything about chess"):
            st.session_state.messages.append({"role": "User", "content": prompt})
            
            # If the user is asking for a move recommendation, include the current board state
            if is_asking_for_move_recommendation(prompt):
                # Create an enhanced prompt that includes the board state
                enhanced_prompt = f"""
                {prompt}
                
                Current board state (FEN): {st.session_state.board.fen()}   
                """
                response = supervisor.invoke({"messages": [HumanMessage(content=enhanced_prompt)]})

                # Only when the turn is white
                if re.match(r'^[^ ]+ w', st.session_state.board.fen()):
                    pgn_moves = get_pgn_format_moves()
                    enhanced_prompt = f"""
                    {prompt}
                    
                    Current board state (FEN): {st.session_state.board.fen()}
    
                    PGN move history: {pgn_moves}
                    """
                    response = supervisor.invoke({"messages": [HumanMessage(content=enhanced_prompt)]})
            else:
                # For general questions, just pass along the original prompt
                response = supervisor.invoke({"messages": [HumanMessage(content=prompt)]})
            
            response_content = response["messages"][-1].content
            st.session_state.messages.append({"role": "Assistant", "content": response_content})
            st.rerun()  # Rerun to update the UI
    
    # Display previous messages in the scrollable container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# ---------- Main App ----------
def main():
    setup_page()
    initialize_state()
    resources = load_resources()
    render_sidebar()
    render_game_area()

if __name__ == "__main__":
    main()