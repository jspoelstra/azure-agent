# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

"""
This module provides an adapter for LangChain tools to be used with Azure Agent.
It allows for the conversion of LangChain tools into functions that can be called
by the Azure Agent, enabling seamless integration with the Azure ecosystem.
"""

import asyncio
import json
from typing import Callable, Dict, Set, List
from langchain_mcp_adapters.client import MultiServerMCPClient

class MCPToolsAdapter:
    """Adapter that converts LangChain tools to functions usable by Azure Agent."""
    
    def __init__(self, mcp_server_url: List[str] = []):
        """Initialize the adapter with MCP server URL."""
        # Store MCP server configuration for later use
        self.mcp_servers = {}
        for ix, url in enumerate(mcp_server_url):
            print(f"Configuring math MCP service at: {url}")
            self.mcp_servers[f"server-{ix}"] = {"url": url, "transport": "sse"}
        
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
