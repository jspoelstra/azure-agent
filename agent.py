# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

"""
DESCRIPTION:
    This sample demonstrates how to use agent operations with toolset from
    the Azure Agents service using a synchronous client.

USAGE:
    python agent-2.py
"""

import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import FunctionTool, ToolSet, CodeInterpreterTool
from user_functions import user_functions
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Agent007:
    def __init__(self):
        # Initialize project client with connection string from environment variable
        self.project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.getenv("PROJECT_CONNECTION_STRING"),
        )
        self.create_agent()
        # Default state is no active conversation
        self.thread = None

    def create_agent(self):
        # Initialize agent toolset with user functions and code interpreter
        # [START create_agent_toolset]
        functions = FunctionTool(user_functions)
        code_interpreter = CodeInterpreterTool()

        toolset = ToolSet()
        toolset.add(functions)
        toolset.add(code_interpreter)

        self.agent = self.project_client.agents.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            name="toolset-agent",
            instructions="You are a helpful assistant",
            toolset=toolset,
        )
        # [END create_agent_toolset]
        print(f"Created agent, ID: {self.agent.id}")

    def create_thread(self):
        # Create thread for communication
        self.thread = self.project_client.agents.create_thread()
        print(f"Created thread, ID: {self.thread.id}")

    def create_message(self, content):
        # Create thread if none exists
        if not self.thread:
            self.create_thread()
        # Create message to thread
        message = self.project_client.agents.create_message(
            thread_id=self.thread.id,
            role="user",
            content=content,
        )
        print(f"Created message, ID: {message.id}")

    def process_message(self, content):
        # Create message to thread
        self.create_message(content)
        # Create and process agent run in thread with tools
        # [START create_and_process_run]
        run = self.project_client.agents.create_and_process_run(
            thread_id=self.thread.id, assistant_id=self.agent.id
        )
        # [END create_and_process_run]
        print(f"Run finished with status: {run.status}")

        if run.status == "failed":
            print(f"Run failed: {run.last_error}")

        # Get and log the last message
        messages = self.project_client.agents.list_messages(thread_id=self.thread.id)
        last_message = messages["data"][0]
        for part in last_message["content"]:
            print(f"AI: {part['text']['value']}")
            print(f"Annotations: {';'.join(part['text']['annotations'])}")
        # Fetch and log all messages
        # messages = self.project_client.agents.list_messages(thread_id=self.thread.id)
        # print(f"Messages: {messages}")

    def delete_agent(self):
        # Delete the assistant when done
        self.project_client.agents.delete_agent(self.agent.id)
        print("Deleted agent")

    def end_conversation(self):
        # Delete thread if it exists
        if self.thread:
            self.project_client.agents.delete_thread(self.thread.id)
            print("Deleted thread")
        self.thread = None


if __name__ == "__main__":
    agent = Agent007()
    # Loop, asking for user input
    while True:
        user_input = input("Enter a message, or 'exit' to quit: ")
        if user_input == "exit":
            break
        agent.process_message(user_input)
    agent.delete_agent()
    agent.end_conversation()
