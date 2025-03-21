# Azure Agent

A collection of samples demonstrating how to use the Azure AI Agent service with various toolsets.

## Description

This repository contains examples of building AI agents using the Azure Agents service. The samples demonstrate how to:
- Create and use agents with function-based tooling
- Implement class-based function tools
- Integrate with pandas and numpy for data processing
- Connect to a MCP (Model Context Protocol) service

## Quick Start

### Prerequisites

- Python 3.8+
- Azure AI Foundry account and project
- For MCP demos: Access to a running MCP server

### Environment Setup

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (create a `.env` file):
```
# Required for all samples
PROJECT_CONNECTION_STRING=your_project_connection_string
MODEL_DEPLOYMENT_NAME=your_model_deployment_name

# Required only for MCP demos
MCP_SERVER_URL=http://your-mcp-server:5001/sse
```

Where:
- `PROJECT_CONNECTION_STRING` - The connection string from your Azure AI Foundry project (found on the overview page)
- `MODEL_DEPLOYMENT_NAME` - The deployment name of the AI model (found under the "Models + endpoints" tab)
- `MCP_SERVER_URL` - The URL to your MCP math service (only needed for agent-mcp.py)

## Running the Samples

The repository contains several sample agent implementations:

### Basic Agent with Function Tools

```bash
python agent.py
```
This sample demonstrates a basic agent implementation with multiple tools from a set of functions.

### File Search Enabled Agent

```bash
python agent-2.py
```
This sample adds file search capability through a vector store.

### Class-based Function Tools

```bash
python agent-3.py
```
This sample demonstrates how to organize function tools within a class.

### Data Processing with Pandas

```bash
python agent-4.py
```
This sample shows integration with pandas and numpy for data processing.

### MCP Tools Integration

```bash
python agent-mcp.py
```
This sample demonstrates integration with Model Context Protocol tools using the LangChain tools client.

**Note:** To run the MCP example, you must have access to a running MCP server and set the `MCP_SERVER_URL` environment variable.

### Testing MCP Connection

To verify your MCP server connection:

```bash
python test_mcp.py
```

## Project Structure

- `agent.py`, `agent-2.py`, etc. - Different agent implementation examples
- `user_functions.py`, `functions2.py`, etc. - Functions that can be used by the agents
- `mcp_tools_adapter.py` - Adapter for integrating LangChain MCP tools
- `test_mcp.py` - Script for testing MCP server connectivity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



