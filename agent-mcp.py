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
    
    def __init__(self):
        # Store MCP server configuration for later use
        math_url = os.getenv("MCP_MATH_URL", "http://localhost:5001/sse")
        self.mcp_servers = {}
        if math_url:
            print(f"Configuring math MCP service at: {math_url}")
            self.mcp_servers["math"] = {"url": math_url, "transport": "sse"}
        
        # Instead of storing a client, store tool definitions
        self.tool_definitions = self._get_tool_definitions()
        
        # Use a dict instead of a set to ensure we don't have name conflicts
        self.functions: Dict[str, Callable] = {}
        self._create_function_wrappers()
    
    def _get_tool_definitions(self):
        """Get tool definitions from MCP server."""
        tool_definitions = []
        
        if not self.mcp_servers:
            print("No MCP servers configured.")
            return tool_definitions
            
        # Use an async function to get tool definitions
        async def get_tools_async():
            try:
                async with MultiServerMCPClient(self.mcp_servers) as mcp_client:
                    tools = mcp_client.get_tools()
                    # Extract minimal info needed to create wrappers
                    for tool in tools:
                        tool_def = {
                            "name": tool.name,
                            "description": tool.description,
                            "schema": None
                        }
                        
                        # Try to get schema information
                        if hasattr(tool, 'args_schema'):
                            try:
                                if isinstance(tool.args_schema, dict) and 'properties' in tool.args_schema:
                                    tool_def["schema"] = tool.args_schema
                                elif hasattr(tool.args_schema, 'schema'):
                                    tool_def["schema"] = tool.args_schema.schema()
                            except Exception as e:
                                print(f"Error extracting schema for {tool.name}: {e}")
                        
                        tool_definitions.append(tool_def)
                    print(f"Found {len(tools)} tools on MCP server")
                    return tool_definitions
            except Exception as e:
                print(f"Error getting MCP tool definitions: {e}")
                return []
                
        # Run the async function to get tool definitions
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tool_definitions = loop.run_until_complete(get_tools_async())
            loop.close()
        except Exception as e:
            print(f"Error in event loop while getting MCP tool definitions: {e}")
        
        return tool_definitions
    
    def _create_function_wrappers(self):
        """Create function wrappers for each MCP tool."""
        for tool_def in self.tool_definitions:
            tool_name = tool_def["name"]
            tool_desc = tool_def["description"]
            schema_dict = tool_def["schema"]
            
            print(f"Creating wrapper for MCP tool: {tool_name}")
            
            # Extract parameter information from schema
            param_info = {}
            if schema_dict and 'properties' in schema_dict:
                param_info = schema_dict['properties']
                print(f"Using schema params for {tool_name}: {list(param_info.keys())}")
                
                # Add required info if available
                if 'required' in schema_dict:
                    for param_name in param_info:
                        param_info[param_name]['required'] = param_name in schema_dict['required']
            
            # Function template with dynamic parameter handling
            def create_wrapper(t_name, t_desc, params):
                # Define parameter annotations to help Azure Agent understand the function signature
                param_annotations = {}
                for param_name in params:
                    param_type = params[param_name].get('type', 'any').lower()
                    if param_type in ('integer', 'number'):
                        param_annotations[param_name] = int if param_type == 'integer' else float
                    elif param_type == 'boolean':
                        param_annotations[param_name] = bool
                    else:
                        param_annotations[param_name] = str
                
                # Create a wrapper that will invoke the MCP tool with a fresh client each time
                def wrapper_function(*args, **kwargs):
                    print(f"Calling MCP tool: {t_name} with args: {kwargs}")
                    
                    if not self.mcp_servers:
                        error_msg = f"Error: No MCP servers configured"
                        print(error_msg)
                        return json.dumps({"error": error_msg, "result": "MCP server not configured"})
                    
                    # Create an async function to run the tool with a fresh client
                    async def run_mcp_tool_async():
                        try:
                            # Import time here to avoid issues with function serialization
                            import time
                            
                            print(f"Creating new MCP client for {t_name} call")
                            async with MultiServerMCPClient(self.mcp_servers) as mcp_client:
                                # Get all tools from the client
                                tools = mcp_client.get_tools()
                                
                                # Find the specific tool we want to use
                                tool = next((t for t in tools if t.name == t_name), None)
                                if tool is None:
                                    raise ValueError(f"Tool {t_name} not found on MCP server")
                                
                                print(f"Invoking {t_name} with params: {kwargs}")
                                start_time = time.time()
                                result = await tool.ainvoke(kwargs)
                                end_time = time.time()
                                print(f"{t_name} completed in {end_time - start_time:.2f}s with result: {result}")
                                return result
                        except Exception as e:
                            print(f"Error in MCP tool execution: {e}")
                            raise
                    
                    # Run the async function with a timeout
                    try:
                        # Create a new event loop for this specific invocation
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        try:
                            # Use a reasonable timeout (10 seconds)
                            result = loop.run_until_complete(
                                asyncio.wait_for(run_mcp_tool_async(), timeout=10.0)
                            )
                            
                            # Return result as JSON string for Azure Agent compatibility
                            if isinstance(result, dict):
                                return json.dumps(result)
                            else:
                                return json.dumps({"result": result})
                        
                        finally:
                            # Clean up the event loop
                            loop.close()
                            
                    except asyncio.TimeoutError:
                        error_msg = f"Error: MCP {t_name} tool timed out after 10 seconds"
                        print(error_msg)
                        return json.dumps({"error": error_msg, "result": "Calculation timed out. Please check if the MCP server is running and accessible."})
                    except Exception as e:
                        error_msg = f"Error invoking {t_name} tool: {str(e)}"
                        print(error_msg)
                        return json.dumps({"error": error_msg, "result": f"Error performing {t_name} operation. Please try again."})
                
                # Set function signature to match parameters
                import inspect
                from inspect import Parameter, Signature
                
                # Create parameters for the function signature
                parameters = []
                for name, details in params.items():
                    # Set default to None so parameters can be called by name
                    param = Parameter(name, Parameter.POSITIONAL_OR_KEYWORD, 
                                      default=None, 
                                      annotation=param_annotations.get(name, inspect.Parameter.empty))
                    parameters.append(param)
                
                # Apply the signature to our wrapper function
                wrapper_function.__signature__ = Signature(parameters)
                
                # Set function metadata
                wrapper_function.__name__ = f"mcp_{t_name}"
                wrapper_function.__qualname__ = f"mcp_{t_name}"
                wrapper_function.__module__ = __name__
                wrapper_function.__annotations__ = param_annotations
                
                # Create a detailed docstring with parameters
                docstring = f"{t_desc} (MCP tool)\n\n"
                for param_name, param_details in params.items():
                    param_type = param_details.get('type', 'any')
                    param_desc = param_details.get('description', f"Parameter for {t_name}")
                    docstring += f":param {param_name}: ({param_type}) {param_desc}\n"
                docstring += f":return: Result from the {t_name} operation"
                
                wrapper_function.__doc__ = docstring
                
                # Create a properly formatted OpenAPI schema for the function
                # This is critical for Azure Agent to understand the function correctly
                wrapper_function.schema = {
                    "name": f"mcp_{t_name}",
                    "description": t_desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param_name: {
                                "type": param_details.get("type", "string").lower().replace("integer", "number"),
                                "description": param_details.get("description", f"Parameter {param_name} for {t_name}")
                            }
                            for param_name, param_details in params.items()
                        },
                        "required": [param_name for param_name, param_details in params.items() 
                                   if param_details.get("required", False)]
                    }
                }
                
                return wrapper_function
            
            # Create a unique wrapper for this tool
            new_wrapper = create_wrapper(tool_name, tool_desc, param_info)
            self.functions[f"mcp_{tool_name}"] = new_wrapper
            print(f"Added dynamic MCP tool wrapper: mcp_{tool_name}")
    
    def get_functions(self) -> Set[Callable]:
        """Return the set of function wrappers."""
        print(f"Returning {len(self.functions)} MCP tool functions")
        return set(self.functions.values())


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
        self.general_functions = Functions4()
        
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
            mcp_adapter = MCPToolsAdapter()
            
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
