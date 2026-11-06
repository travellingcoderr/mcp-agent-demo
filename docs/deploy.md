# Deploying the demo

The container serves the UI at `/`, the API at `POST /api/ask` and a health check at `/healthz` on port 8080. It needs outbound HTTPS to `api.fda.gov` and to your LLM endpoint.

Environment:

| Variable | Required | Example |
|---|---|---|
| `LLM_BASE_URL` | yes | `https://api.openai.com/v1`, or an Azure OpenAI / vLLM OpenAI-compatible URL |
| `LLM_MODEL` | yes | any model with tool calling |
| `OPENAI_API_KEY` | yes (store as a secret) | |
| `OPENFDA_API_KEY` | no | raises the openFDA rate limit |

## Azure Container Apps

```bash
az group create -n drug-agent -l eastus
az acr create -g drug-agent -n <acrname> --sku Basic
az acr build -r <acrname> -t drug-safety-agent:1 .
az containerapp up -g drug-agent -n drug-safety-agent \
  --image <acrname>.azurecr.io/drug-safety-agent:1 --ingress external --target-port 8080 \
  --env-vars LLM_BASE_URL=... LLM_MODEL=... OPENAI_API_KEY=secretref:llm-key
az containerapp secret set -g drug-agent -n drug-safety-agent --secrets llm-key=<key>
```

Set min replicas to 0 to pay nothing while idle (first request then takes a few seconds).

## AWS App Runner

Push the image to ECR, then create an App Runner service from it with port 8080, health check path `/healthz`, and the variables above (put the API key in Secrets Manager and reference it). App Runner gives you an HTTPS URL directly.

## Before making it public

- Put a rate limit in front (both services support it, or use the gateway from llm-platform-k8s) so the link can't run up your LLM bill.
- Run the red-team gate from ai-redteam-toolkit against `/api/ask`.
