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
        """Create function wrappers for each LangChain tool dynamically."""
        for lc_tool in self.lc_tools:
            # Store tool reference and details to avoid closure issues
            tool_ref = lc_tool
            tool_name = lc_tool.name
            tool_desc = lc_tool.description
            print(f"Creating wrapper for MCP tool: {tool_name}")
            
            # Extract parameter information when possible
            param_info = {}
            try:
                if hasattr(tool_ref, 'args_schema'):
                    # Handle different ways tools might structure their schema
                    schema_dict = None
                    
                    # Case 1: args_schema is already a dict with properties
                    if isinstance(tool_ref.args_schema, dict) and 'properties' in tool_ref.args_schema:
                        schema_dict = tool_ref.args_schema
                        print(f"Found direct schema dict for {tool_name}")
                    
                    # Case 2: args_schema has schema() method (Pydantic model)
                    elif hasattr(tool_ref.args_schema, 'schema'):
                        schema_dict = tool_ref.args_schema.schema()
                        print(f"Found schema method for {tool_name}")
                    
                    # Case 3: args_schema is the schema
                    elif hasattr(tool_ref.args_schema, '__fields__') or hasattr(tool_ref.args_schema, 'model_fields'):
                        # This might be a direct Pydantic model, try to extract info directly
                        field_dict = getattr(tool_ref.args_schema, '__fields__', None) or getattr(tool_ref.args_schema, 'model_fields', {})
                        param_info = {
                            name: {
                                'type': getattr(field, 'type_', str(field.annotation)) if hasattr(field, 'annotation') else 'any',
                                'description': getattr(field, 'description', f"Parameter {name} for {tool_name}")
                            }
                            for name, field in field_dict.items()
                        }
                        print(f"Extracted params directly from model for {tool_name}: {list(param_info.keys())}")
                    
                    # Extract properties from schema dict if we have one
                    if schema_dict and 'properties' in schema_dict:
                        param_info = schema_dict['properties']
                        print(f"Discovered parameters for {tool_name}: {list(param_info.keys())}")
                        
                        # Add required info if available
                        if 'required' in schema_dict:
                            for param_name in param_info:
                                param_info[param_name]['required'] = param_name in schema_dict['required']
                        
                        # Add title info if available
                        for param_name, param_details in param_info.items():
                            if 'title' in param_details:
                                # If no description, use title as description
                                if 'description' not in param_details:
                                    param_info[param_name]['description'] = param_details['title']
                
                # If no schema found but tool has annotations, use those
                if not param_info and hasattr(tool_ref, '__annotations__'):
                    param_info = {name: {'type': str(typ)} for name, typ in tool_ref.__annotations__.items()}
                
                # Extract parameter descriptions from docstring if available
                if hasattr(tool_ref, '__doc__') and tool_ref.__doc__:
                    doc_lines = tool_ref.__doc__.split('\n')
                    for line in doc_lines:
                        line = line.strip()
                        # Check for parameter documentation patterns like ":param name: description"
                        if line.startswith(':param ') or line.startswith('@param '):
                            parts = line.split(':', 2) if ':' in line else line.split(' ', 2)
                            if len(parts) >= 3:
                                param_name = parts[1].strip()
                                param_desc = parts[2].strip()
                                if param_name in param_info:
                                    param_info[param_name]['description'] = param_desc
                
                # Add detailed debug of tool schema for troubleshooting
                print(f"Final parameter info for {tool_name}: {param_info}")
                
                # For commonly known tools, provide fallback params with descriptions
                if not param_info and tool_name in ['add', 'multiply', 'divide']:
                    param_info = {
                        'a': {'type': 'number', 'description': 'First number to operate on'},
                        'b': {'type': 'number', 'description': 'Second number to operate on'}
                    }
                
            except Exception as e:
                print(f"Error extracting parameters for {tool_name}: {e}")
            
            # Function template with dynamic parameter handling
            def create_wrapper(t_ref, t_name, t_desc, params):
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
                
                # Simpler approach - create a closure that captures the tool reference
                def wrapper_function(*args, **kwargs):
                    print(f"Calling MCP tool: {t_name} with args: {kwargs}")
                    
                    try:
                        # Pass parameters directly without type conversion
                        # This preserves the original parameter types as provided by the LLM
                        print(f"Using parameters for {t_name}: {kwargs}")
                        
                        # Super simple approach - directly use asyncio.run() for each call
                        async def simple_invoke():
                            start_time = time.time()
                            print(f"Starting direct async call to {t_name}...")
                            # Pass parameters directly without modification
                            result = await t_ref.ainvoke(kwargs)
                            end_time = time.time()
                            print(f"{t_name} call completed successfully in {end_time - start_time:.2f} seconds with result: {result}")
                            return result
                        
                        # Use asyncio.run() with shorter timeout
                        print(f"Invoking {t_name} using direct asyncio.run() approach")
                        result = asyncio.run(asyncio.wait_for(simple_invoke(), timeout=5.0))
                        
                        # Return result as JSON string for Azure Agent compatibility
                        if isinstance(result, dict):
                            return json.dumps(result)
                        else:
                            return json.dumps({"result": result})
                            
                    except asyncio.TimeoutError:
                        error_msg = f"Error: MCP {t_name} tool timed out after 5 seconds"
                        print(error_msg)
                        return json.dumps({"error": error_msg, "result": f"Calculation timed out. Please check if the MCP server is running and accessible."})
                    except Exception as e:
                        error_msg = f"Error invoking {t_name} tool: {str(e)}"
                        print(error_msg)
                        return json.dumps({"error": error_msg, "result": f"Error performing {t_name} operation. Please try again."})
                
                # Set function signature to match parameters
                import inspect
                import time  # Add time module for timing measurements
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
                
                # Debug the function signature and parameters as seen by Azure Agent
                print(f"\n--- Function Signature Debug for mcp_{t_name} ---")
                print(f"Function name: mcp_{t_name}")
                print(f"Function description: {t_desc}")
                print(f"Parameters schema: {json.dumps(params, indent=2)}")
                print(f"Function annotations: {wrapper_function.__annotations__}")
                print(f"Function signature: {wrapper_function.__signature__}")
                print(f"Function schema: {json.dumps(wrapper_function.schema, indent=2)}\n")
                
                return wrapper_function
            
            # Create a unique wrapper for this tool
            new_wrapper = create_wrapper(tool_ref, tool_name, tool_desc, param_info)
            self.functions[f"mcp_{tool_name}"] = new_wrapper
            print(f"Added dynamic MCP tool wrapper: mcp_{tool_name}")
    
    def get_functions(self) -> Set[Callable]:
        """Return the set of function wrappers."""
        print(f"Returning {len(self.functions)} MCP tool functions")
        return set(self.functions.values())
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'loop') and self.loop and not self.loop.is_closed():
            self.loop.close()


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
        
        # Initialize MCP client with service URLs from environment variables
        math_url = os.getenv("MCP_MATH_URL", "http://20.7.111.111:5001/sse")
        
        # Define servers dictionary with available services
        mcp_servers = {}
        
        # Only add services with valid URLs
        if math_url:
            print(f"Adding math MCP service at: {math_url}")
            mcp_servers["math"] = {"url": math_url, "transport": "sse"}
                
        # Create MCP client with discovered services
        self.mcp_client = MultiServerMCPClient(mcp_servers)
        self.mcp_adapter = None
        
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
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Manually enter the context manager
            loop.run_until_complete(self.mcp_client.__aenter__())
            
            # Create combined set of functions
            all_functions = set()
            
            # Create adapter for MCP tools and get functions
            print("Creating MCP tools adapter...")
            self.mcp_adapter = MCPToolsAdapter(self.mcp_client)
            
            # Add MCP functions to the combined set
            if self.mcp_adapter.functions:
                print("Adding MCP functions...")
                all_functions.update(self.mcp_adapter.get_functions())
            
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
            # Clean up resources if there was an error
            if hasattr(self, 'mcp_adapter') and self.mcp_adapter:
                self.mcp_adapter.cleanup()
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
            
            # Clean up MCP adapter resources
            if hasattr(self, 'mcp_adapter') and self.mcp_adapter:
                self.mcp_adapter.cleanup()
            
            # Close MCP client
            if hasattr(self, 'mcp_client') and self.mcp_client:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    try:
                        loop.run_until_complete(self.mcp_client.__aexit__(None, None, None))
                    except Exception as e:
                        print(f"Error closing MCP client: {str(e)}")
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
