#!/usr/bin/env python3
"""
Simple test script to diagnose MCP server connectivity issues.
This bypasses the Azure Agent and directly tests the MCP connection.
"""

import os
import sys
import asyncio
import json
import time
from typing import Dict

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
load_dotenv()

async def test_mcp_connection():
    """Test connecting to the MCP server and invoking a simple function."""
    
    # Get MCP server URL from environment or use default
    math_url = os.getenv("MCP_MATH_URL", "http://20.7.111.111:5001/sse")
    print(f"Testing connection to MCP server at: {math_url}")
    
    # Configure the MCP client
    mcp_servers = {"math": {"url": math_url, "transport": "sse"}}
    
    try:
        # Create and enter the MCP client context
        async with MultiServerMCPClient(mcp_servers) as mcp_client:
            # Get available tools
            tools = mcp_client.get_tools()
            print(f"Found {len(tools)} tools on the MCP server")
            
            # List the available tools
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Try to use the multiply tool if available
            multiply_tool = next((tool for tool in tools if tool.name == "multiply"), None)
            if multiply_tool:
                print("\nTesting the multiply tool...")
                try:
                    # Test with timeout
                    print("Test 1: With 10-second timeout")
                    start_time = time.time()
                    result = await asyncio.wait_for(
                        multiply_tool.ainvoke({"a": 3.0, "b": 4.0}),
                        timeout=10.0
                    )
                    end_time = time.time()
                    print(f"Success! Result: {result} (took {end_time - start_time:.2f} seconds)")
                    
                    # Test without timeout, but with different values
                    print("\nTest 2: Without explicit timeout")
                    start_time = time.time()
                    result = await multiply_tool.ainvoke({"a": 5.0, "b": 6.0})
                    end_time = time.time()
                    print(f"Success! Result: {result} (took {end_time - start_time:.2f} seconds)")
                    
                    # Test with larger numbers
                    print("\nTest 3: With larger numbers")
                    start_time = time.time()
                    result = await multiply_tool.ainvoke({"a": 15.0, "b": 16.0})
                    end_time = time.time()
                    print(f"Success! Result: {result} (took {end_time - start_time:.2f} seconds)")
                    
                except asyncio.TimeoutError:
                    print("Error: Operation timed out after 10 seconds")
                except Exception as e:
                    print(f"Error invoking multiply tool: {str(e)}")
            else:
                print("Multiply tool not found on the server")
    
    except Exception as e:
        print(f"Error connecting to MCP server: {str(e)}")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_mcp_connection())