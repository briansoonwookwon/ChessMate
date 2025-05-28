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