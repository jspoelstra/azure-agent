# azure-agent
Test of the Azure Agent Service

## DESCRIPTION:
This repo modifies a Microsoft sample that demonstrates how to use agent operations with a toolset from the Azure Agents service using a synchronous client.

## Quick Start:

### Environment
Before running the sample:
- `python -m venv venv`
- `source venv/bin/activate`  # for Linux/macOS
- `.\venv\Scripts\activate`  # for Windows
- `pip install -r requirements.txt`

Set these environment variables with your own values:
1. `PROJECT_CONNECTION_STRING` - The project connection string, as found in the overview page of your Azure AI Foundry project.
2. `MODEL_DEPLOYMENT_NAME` - The deployment name of the AI model, as found under the "Name"column in the "Models + endpoints" tab in your Azure AI Foundry project.

### Running the sample
    python agent.py



