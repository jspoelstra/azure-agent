# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

"""
DESCRIPTION:
    This sample demonstrates how to use MCP tools with Azure Agent.

USAGE:
    python agent-mcp.py
"""

import os
import asyncio
import json
import inspect
from typing import Any, Callable, Dict, List, Set, Optional
from functools import wraps

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import (
    FunctionTool,
    ToolSet,
    CodeInterpreterTool,
    FileSearchTool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from mcp_tools_adapter import MCPToolsAdapter

# Load environment variables from .env file
load_dotenv()

class Agent007:
    def __init__(self):
        self.agent_name = "agent-mcp"
        self.instructions = """
        You are a helpful assistant with access to special tools.
        You have access to MCP tools for various operations.
        
        MCP tools have names starting with "mcp_" followed by the operation name.
        For example: "mcp_add", "mcp_multiply", etc.
                
        If a tool call times out or fails, inform the user and suggest they check if the server is running.
        """
        # Initialize project client with connection string from environment variable
        self.project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.getenv("PROJECT_CONNECTION_STRING"),
        )
        self.create_file_search_tool()
        
        # Initialize agent toolset with user functions and code interpreter
        self.create_agent()
        # Default state is no active conversation
        self.thread = None

    def create_file_search_tool(self):
        # Get list of existing vector stores
        vector_stores = self.project_client.agents.list_vector_stores()
        # Create file search tool
        try:
            self.file_search_tool = FileSearchTool(vector_store_ids=[vector_stores.first_id])
        except:
            print("No vector stores found, file search tool will not be available")
            self.file_search_tool = None

    def create_agent(self):
        # Initialize agent toolset with user functions and code interpreter
        toolset = ToolSet()
        
        try:
            # Create combined set of functions
            all_functions = set()
            
            # Create adapter for MCP tools and get functions
            print("Creating MCP tools adapter...")
            mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:5001/sse")
            mcp_adapter = MCPToolsAdapter([mcp_server_url])
            
            # Add MCP functions to the combined set
            if mcp_adapter.functions:
                print("Adding MCP functions...")
                all_functions.update(mcp_adapter.get_functions())
                        
            # Create a single FunctionTool with all the functions
            print(f"Creating combined FunctionTool with {len(all_functions)} functions")
            combined_functions = FunctionTool(all_functions)
            toolset.add(combined_functions)
            
            # Add code interpreter
            code_interpreter = CodeInterpreterTool()
            toolset.add(code_interpreter)
            
            # Add file search tool if available
            if self.file_search_tool:
                toolset.add(self.file_search_tool)

            print("Creating agent...")
            self.agent = self.project_client.agents.create_agent(
                model=os.environ["MODEL_DEPLOYMENT_NAME"],
                name=self.agent_name,
                instructions=self.instructions,
                toolset=toolset,
            )
            print(f"Created agent, ID: {self.agent.id}")
        except Exception as e:
            print(f"Error setting up agent: {str(e)}")
            raise

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
        try:
            # Create message to thread
            self.create_message(content)
            # Create and process agent run in thread with tools
            run = self.project_client.agents.create_and_process_run(
                thread_id=self.thread.id, assistant_id=self.agent.id
            )
            print(f"Run finished with status: {run.status}")

            if run.status == "failed":
                print(f"Run failed: {run.last_error}")

            # Get and log the last message
            messages = self.project_client.agents.list_messages(thread_id=self.thread.id)
            last_message = messages["data"][0]
            for part in last_message["content"]:
                print(f"AI: {part['text']['value']}")
                if hasattr(part, 'text') and hasattr(part.text, 'annotations') and len(part.text.annotations) > 0:
                    print(f"Annotations:")
                    for annotation in part.text.annotations:
                        print(f"  {annotation}")
        except Exception as e:
            print(f"Error processing message: {str(e)}")

    def delete_agent(self):
        try:
            # Delete the assistant when done
            if hasattr(self, 'agent') and self.agent:
                self.project_client.agents.delete_agent(self.agent.id)
                print("Deleted agent")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def end_conversation(self):
        try:
            # Delete thread if it exists
            if self.thread:
                self.project_client.agents.delete_thread(self.thread.id)
                print("Deleted thread")
            self.thread = None
        except Exception as e:
            print(f"Error ending conversation: {str(e)}")


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
