import boto3
import datetime
import os
from pathlib import Path
from botocore.session import get_session
from botocore.credentials import RefreshableCredentials
from langchain_aws import ChatBedrockConverse

# ARN of Role A to assume  
ROLE_TO_ASSUME = Path(os.path.join(os.environ["HOME"],"BedrockCrossAccount.txt")).read_text().strip()


def get_credentials():
    sts_client = boto3.client('sts')
    assumed_role = sts_client.assume_role(
        RoleArn=ROLE_TO_ASSUME,
        RoleSessionName='cross-account-session',
        # Don't set DurationSeconds when role chaining
    )
    return {
        'access_key': assumed_role['Credentials']['AccessKeyId'],
        'secret_key': assumed_role['Credentials']['SecretAccessKey'],
        'token': assumed_role['Credentials']['SessionToken'],
        'expiry_time': assumed_role['Credentials']['Expiration'].isoformat()
    }

session = get_session()
refresh_creds = RefreshableCredentials.create_from_metadata(
    metadata=get_credentials(),
    refresh_using=get_credentials,
    method='sts-assume-role'
)

# Create a new session with refreshable credentials
session._credentials = refresh_creds
boto3_session = boto3.Session(botocore_session=session)

region: str = "us-east-1"

# ---- ⚠️ Update region for your AWS setup ⚠️ ----
bedrock_client = boto3_session.client("bedrock-runtime",
                              region_name=region)

import chess
import chess.pgn
import requests
import json
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

model = init_chat_model("us.amazon.nova-micro-v1:0", #us.anthropic.claude-3-5-haiku-20241022-v1:0"
                        model_provider="bedrock_converse",
                        region_name="us-east-1",
                        client=bedrock_client)

@tool
def fetch_puzzle(query: str) -> dict:
    """
    Get a random puzzle from chess.com in a json format.
    """
    url = "https://api.chess.com/pub/puzzle/random"
    headers = {'User-Agent': 'My Python App'}
    response = requests.get(url, headers=headers).json()

    return response

@tool
def get_info(response: dict) ->  dict:
    """
    Get a FEN from the response on chess.com.
    """
    pgn_io = io.StringIO(response['pgn'])
    game = chess.pgn.read_game(pgn_io)

    info = {
        'title' : response['title'],
        'fen' : response['fen'],
        'answer' : str(game.mainline_moves())
    }
    
    return info

def create_puzzle_agent():
    """
    Create a agent to fetch a chess puzzle from the chess.com
    """
    
    tools = [fetch_puzzle, get_info]
    
    prompt = """
    You are an intelligent agent that helps users to get a chess puzzle.
    Follow these steps to answer user queries:
    1. Use the `fetch_puzzle` tool to request a puzzle using chess.com API.
    2. Use the `get_info` tool to retrieve relevant information of the puzzle.
    4. If you can't find enough information, state that explicitly.
    
    Never include explanations, steps, or any text besides the dictionary info or error message.
    Only provide a dictoinary from the get_info tool.
    """
    
    puzzle_agent = create_react_agent(model, tools, prompt = prompt)

    return puzzle_agent

