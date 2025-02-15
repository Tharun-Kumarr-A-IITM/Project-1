# LLM-based Automation Agent

This repository contains an automation agent built with FastAPI that processes plain-English tasks to automate various file and data operations. It integrates with an LLM (GPT-4o-Mini) via an API call using your provided `AIPROXY_TOKEN`.

## Endpoints

- **POST /run?task=&lt;task description&gt;**  
  Executes a plain‑English task.

- **GET /read?path=&lt;file path&gt;**  
  Returns the content of a specified file (only files under `/data`).

## Setup

1. **Environment Variable:**  
   Set the `AIPROXY_TOKEN` in your environment.
   ```bash
   export AIPROXY_TOKEN=your_actual_token_here
   ```
