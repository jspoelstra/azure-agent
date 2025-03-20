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
from functions4 import Functions4
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class MCPToolsAdapter:
    """Adapter that converts LangChain tools to functions usable by Azure Agent."""
    
    def __init__(self, mcp_client: MultiServerMCPClient):
        self.mcp_client = mcp_client
        self.lc_tools = mcp_client.get_tools()
        print(f"MCP client returned {len(self.lc_tools)} tools")
        # Use a dict instead of a set to ensure we don't have name conflicts
        self.functions: Dict[str, Callable] = {}
        self._create_function_wrappers()
    
    def _create_function_wrappers(self):
        """Create function wrappers for each LangChain tool."""
        for lc_tool in self.lc_tools:
            # Store tool reference in the wrapper's closure
            tool_ref = lc_tool
            tool_name = lc_tool.name
            tool_desc = lc_tool.description
            
            # Create hardcoded parameter handlers based on tool name
            if tool_name == 'add':
                # Create a function that directly handles arguments
                def wrapper_function(a: float, b: float):
                    print(f"Calling MCP add tool with a={a}, b={b}")
                    
                    # Create a new event loop for this function call
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        # Use ainvoke directly with the required parameters with a timeout
                        async def call_with_timeout():
                            return await asyncio.wait_for(
                                tool_ref.ainvoke({"a": float(a), "b": float(b)}),
                                timeout=10.0  # 10 second timeout
                            )
                        
                        print("Executing MCP add tool with timeout...")
                        result = loop.run_until_complete(call_with_timeout())
                        print(f"Add result: {result}")
                        
                        # Return result as JSON string for Azure Agent compatibility
                        if isinstance(result, dict):
                            return json.dumps(result)
                        else:
                            return json.dumps({"result": result})
                    except asyncio.TimeoutError:
                        error_msg = "Error: MCP add tool timed out after 10 seconds"
                        print(error_msg)
                        return json.dumps({"error": error_msg, "result": f"Calculation timed out. Please check if the MCP server at {self.mcp_client.servers['math']['url']} is running and accessible."})
                    except Exception as e:
                        error_msg = f"Error invoking add tool: {str(e)}"
                        print(error_msg)
                        return json.dumps({"error": error_msg, "result": "Error performing calculation. Please try again."})
                    finally:
                        loop.close()
                
                wrapper_function.__name__ = "mcp_add"
                wrapper_function.__doc__ = """Add two numbers (MCP tool)
                
                :param a: (float) First number to add
                :param b: (float) Second number to add
                :return: Result of the addition operation
                """
                self.functions["mcp_add"] = wrapper_function
            
            elif tool_name == 'multiply':
                # Create a function that directly handles arguments
                def wrapper_function(a: float, b: float):
                    print(f"Calling MCP multiply tool with a={a}, b={b}")
                    
                    # Create a new event loop for this function call
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        # Use ainvoke directly with the required parameters with a timeout
                        async def call_with_timeout():
                            print(f"Starting async multiply call with params: a={a}, b={b}")
                            result = await asyncio.wait_for(
                                tool_ref.ainvoke({"a": float(a), "b": float(b)}),
                                timeout=10.0  # 10 second timeout
                            )
                            print(f"Async multiply call completed with result: {result}")
                            return result
                        
                        print("Executing MCP multiply tool with timeout...")
                        result = loop.run_until_complete(call_with_timeout())
                        print(f"Multiply result: {result}")
                        
                        # Return result as JSON string for Azure Agent compatibility
                        if isinstance(result, dict):
                            return json.dumps(result)
                        else:
                            return json.dumps({"result": result})
                    except asyncio.TimeoutError:
                        error_msg = "Error: MCP multiply tool timed out after 10 seconds"
                        print(error_msg)
                        return json.dumps({"error": error_msg, "result": f"Calculation timed out. Please check if the MCP server at {self.mcp_client.servers['math']['url']} is running and accessible."})
                    except Exception as e:
                        error_msg = f"Error invoking multiply tool: {str(e)}"
                        print(error_msg)
                        return json.dumps({"error": error_msg, "result": "Error performing calculation. Please try again."})
                    finally:
                        loop.close()
                
                wrapper_function.__name__ = "mcp_multiply"
                wrapper_function.__doc__ = """Multiply two numbers (MCP tool)
                
                :param a: (float) First number to multiply
                :param b: (float) Second number to multiply
                :return: Result of the multiplication operation
                """
                self.functions["mcp_multiply"] = wrapper_function
            
            else:
                print(f"Unknown tool: {tool_name}, not adding wrapper")
    
    def get_functions(self) -> Set[Callable]:
        """Return the set of function wrappers."""
        print(f"Returning {len(self.functions)} MCP tool functions")
        return set(self.functions.values())


class Agent007:
    def __init__(self):
        self.agent_name = "agent-mcp"
        self.instructions = """
        You are a helpful assistant with access to special tools.
        You have access to math tools that can perform calculations.
        
        For the math tools:
        - mcp_add: adds two numbers (parameters: a and b)
        - mcp_multiply: multiplies two numbers (parameters: a and b)
        
        Always provide numeric values directly as parameters, not as strings.
        For example:
        - CORRECT: Use mcp_multiply with parameters a=5, b=10
        - INCORRECT: Don't use mcp_multiply with parameters kwargs="5,10"
        
        If a tool call times out or fails, inform the user and suggest they check if the server is running.
        """
        # Initialize project client with connection string from environment variable
        self.project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.getenv("PROJECT_CONNECTION_STRING"),
        )
        self.create_file_search_tool()
        self.general_functions = Functions4()
        
        # Initialize MCP client with the math service - make sure it's running 
        # Use fallback URLs for testing
        math_url = os.getenv("MCP_MATH_URL", "http://20.22.134.111:5001/sse")
        
        print(f"Connecting to MCP math service at: {math_url}")
        self.mcp_client = MultiServerMCPClient(
            {
                "math": {
                    "url": math_url,
                    "transport": "sse",
                }
            }
        )
        
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
        
        # Start MCP client and get tools
        print("Starting MCP client...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Manually enter the context manager
        loop.run_until_complete(self.mcp_client.__aenter__())
        
        # Create combined set of functions
        all_functions = set()
        
        # Create adapter for MCP tools and get functions
        print("Creating MCP tools adapter...")
        mcp_adapter = MCPToolsAdapter(self.mcp_client)
        
        # Add MCP functions to the combined set
        if mcp_adapter.functions:
            print("Adding MCP functions...")
            all_functions.update(mcp_adapter.get_functions())
        
        # Add general functions to the combined set
        print("Adding general functions...")
        all_functions.update(self.general_functions.user_functions)
        
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

    def delete_agent(self):
        # Delete the assistant when done
        self.project_client.agents.delete_agent(self.agent.id)
        print("Deleted agent")
        
        # Close MCP client
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.run_until_complete(self.mcp_client.__aexit__(None, None, None))

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
