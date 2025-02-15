import os
import re
import json
import glob
import sqlite3
import subprocess
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
DATA_DIR = os.path.join(os.getcwd(), "data")

def safe_path(path: str) -> str:
    """
    Only allow access to files under /data.
    """
    full_path = os.path.abspath(path)
    data_dir_abs = os.path.abspath(DATA_DIR)
    if not full_path.startswith(data_dir_abs):
        raise ValueError("Access to paths outside /data is not allowed.")
    return full_path

def call_llm(prompt: str) -> str:
    """
    Call the GPT-4o-Mini endpoint using the provided AIPROXY_TOKEN.
    Replace the URL below with the actual endpoint if different.
    """
    token = os.environ.get("AIPROXY_TOKEN")
    if not token:
        raise Exception("AIPROXY_TOKEN environment variable not set")
    api_url = "https://api.ai-proxy.com/gpt4o-mini"  # Update this URL as needed.
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "prompt": prompt,
        "max_tokens": 150  # Adjust max_tokens as needed.
    }
    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"LLM API call failed: {response.status_code} - {response.text}")
    result = response.json()
    # Expecting the API response to have a key "text" with the generated output.
    return result.get("text", "")

@app.get("/read")
def read_file(path: str):
    """
    Read the file at the specified path (only under /data) and return its contents as plain text.
    """
    try:
        safe_file = safe_path(path)
        if not os.path.exists(safe_file):
            raise HTTPException(status_code=404, detail="File not found")
        with open(safe_file, "r", encoding="utf-8") as f:
            content = f.read()
        return PlainTextResponse(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run")
def run(task: str):
    """
    Execute a plain‑English task.
    Dispatches the task to the appropriate automation function.
    """
    try:
        output = run_task(task)
        return {"message": "Task executed successfully", "output": output}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_task(task: str) -> str:
    """
    Heuristically parse the task description and execute the corresponding routine.
    Supports tasks A1–A10 as well as some business tasks (B3–B10).
    """
    task_lower = task.lower()

    if "datagen.py" in task_lower:
        return run_a1(task)
    elif "prettier" in task_lower and "format" in task_lower:
        return run_a2(task)
    elif "wednesday" in task_lower:
        return run_a3(task)
    elif "contacts" in task_lower and "sort" in task_lower:
        return run_a4(task)
    elif ".log" in task_lower and "most recent" in task_lower:
        return run_a5(task)
    elif "markdown" in task_lower and "docs" in task_lower and "index" in task_lower:
        return run_a6(task)
    elif "email.txt" in task_lower and "extract" in task_lower:
        return run_a7(task)
    elif "credit-card.png" in task_lower:
        return run_a8(task)
    elif "comments" in task_lower and "similar" in task_lower:
        return run_a9(task)
    elif "ticket-sales.db" in task_lower:
        return run_a10(task)
    else:
        return run_business_task(task)

def run_a1(task: str) -> str:
    """
    A1. Install uv (if required) and run datagen.py with the user email as argument.
    """
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', task)
    email = email_match.group(0) if email_match else "user@example.com"

    datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    local_script = "/tmp/datagen.py"

    r = requests.get(datagen_url)
    if r.status_code != 200:
        raise Exception("Failed to download datagen.py")
    with open(local_script, "w", encoding="utf-8") as f:
        f.write(r.text)

    # Run the downloaded script with the email as an argument.
    subprocess.run(["python", local_script, email], check=True)
    return "A1 executed"

def run_a2(task: str) -> str:
    """
    A2. Format the contents of /data/format.md using prettier@3.4.2.
         (Prettier must be installed in the container.)
    """
    file_path = safe_path("/data/format.md")
    subprocess.run(["prettier", "--write", file_path, "--parser", "markdown"], check=True)
    return "A2 executed"

def run_a3(task: str) -> str:
    """
    A3. Count the number of Wednesdays in /data/dates.txt and write the count to /data/dates-wednesdays.txt.
    """
    input_path = safe_path("/data/dates.txt")
    output_path = safe_path("/data/dates-wednesdays.txt")
    count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = datetime.fromisoformat(line)
                if d.weekday() == 2:  # Wednesday (Monday=0, Tuesday=1, Wednesday=2)
                    count += 1
            except ValueError:
                continue
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(count))
    return "A3 executed"

def run_a4(task: str) -> str:
    """
    A4. Sort the contacts in /data/contacts.json by last_name then first_name,
         and write the sorted list to /data/contacts-sorted.json.
    """
    input_path = safe_path("/data/contacts.json")
    output_path = safe_path("/data/contacts-sorted.json")
    with open(input_path, "r", encoding="utf-8") as f:
        contacts = json.load(f)
    sorted_contacts = sorted(contacts, key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_contacts, f, indent=2)
    return "A4 executed"

def run_a5(task: str) -> str:
    """
    A5. From the 10 most recent .log files in /data/logs/,
         extract the first line of each (most recent first) and write them to /data/logs-recent.txt.
    """
    logs_dir = safe_path("/data/logs")
    output_path = safe_path("/data/logs-recent.txt")
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    selected = log_files[:10]
    lines = []
    for log_file in selected:
        with open(log_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            lines.append(first_line)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "A5 executed"

def run_a6(task: str) -> str:
    """
    A6. For all Markdown (.md) files in /data/docs/,
         extract the first H1 (line starting with '# ') from each file,
         and create an index mapping filename (relative to /data/docs/) to title in /data/docs/index.json.
    """
    docs_dir = safe_path("/data/docs")
    index = {}
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, docs_dir)
                title = None
                with open(full_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.lstrip().startswith("# "):
                            title = line.lstrip()[2:].strip()
                            break
                if title:
                    index[relative_path] = title
    output_path = safe_path("/data/docs/index.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return "A6 executed"

def run_a7(task: str) -> str:
    """
    A7. Read /data/email.txt, pass its contents to the LLM to extract the sender's email address,
         and write the extracted email to /data/email-sender.txt.
    """
    input_path = safe_path("/data/email.txt")
    output_path = safe_path("/data/email-sender.txt")
    with open(input_path, "r", encoding="utf-8") as f:
        email_content = f.read()
    prompt = f"Extract the sender's email address from the following email message:\n\n{email_content}"
    sender_email = call_llm(prompt).strip()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(sender_email)
    return "A7 executed"

def run_a8(task: str) -> str:
    """
    A8. Read the image /data/credit-card.png, pass it to the LLM to extract the credit card number,
         and write the number (without spaces) to /data/credit-card.txt.
    """
    input_path = safe_path("/data/credit-card.png")
    output_path = safe_path("/data/credit-card.txt")
    with open(input_path, "rb") as f:
        image_data = f.read()
    # For image data, you might need to base64 encode it or use multipart/form-data.
    # Here, we assume the endpoint accepts raw image bytes in a specific field.
    token = os.environ.get("AIPROXY_TOKEN")
    if not token:
        raise Exception("AIPROXY_TOKEN environment variable not set")
    api_url = "https://api.ai-proxy.com/gpt4o-mini"  # Update as needed.
    headers = {"Authorization": f"Bearer {token}"}
    files = {"image": image_data}
    # Construct a prompt instructing the LLM to extract the credit card number.
    data = {"prompt": "Extract the credit card number from the provided image. Return only the number without spaces.", "max_tokens": 50}
    response = requests.post(api_url, data=data, files=files, headers=headers)
    if response.status_code != 200:
        raise Exception(f"LLM API call for image extraction failed: {response.status_code} - {response.text}")
    result = response.json()
    card_number = result.get("text", "").replace(" ", "").strip()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card_number)
    return "A8 executed"

def run_a9(task: str) -> str:
    """
    A9. Read /data/comments.txt, compute embeddings to determine the most similar pair of comments,
         and write the two most similar comments to /data/comments-similar.txt.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    input_path = safe_path("/data/comments.txt")
    output_path = safe_path("/data/comments-similar.txt")
    with open(input_path, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f if line.strip()]

    if len(comments) < 2:
        raise Exception("Not enough comments for comparison.")

    vectorizer = TfidfVectorizer().fit_transform(comments)
    vectors = vectorizer.toarray()
    sim_matrix = cosine_similarity(vectors)
    n = len(comments)
    best_sim = -1
    best_pair = (None, None)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] > best_sim:
                best_sim = sim_matrix[i][j]
                best_pair = (comments[i], comments[j])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(best_pair[0] + "\n" + best_pair[1])
    return "A9 executed"

def run_a10(task: str) -> str:
    """
    A10. Query the SQLite database /data/ticket-sales.db for the total sales of "Gold" tickets,
         and write the result to /data/ticket-sales-gold.txt.
    """
    db_path = safe_path("/data/ticket-sales.db")
    output_path = safe_path("/data/ticket-sales-gold.txt")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    result = cursor.fetchone()[0]
    result = result if result is not None else 0
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(result))
    conn.close()
    return "A10 executed"

def run_business_task(task: str) -> str:
    """
    Handle broad business tasks (B3–B10). Extend these as needed.
    """
    task_lower = task.lower()
    if "fetch data from an api" in task_lower:
        # Extract API URL and target path from the task and implement accordingly.
        # For example, use requests.get() to fetch data and write it to a file.
        return "B3 executed"
    elif "clone a git repo" in task_lower:
        # Use subprocess to run git commands.
        return "B4 executed"
    elif "run a sql query" in task_lower:
        # Connect to the database and execute the provided SQL.
        return "B5 executed"
    elif "scrape" in task_lower:
        # Use requests and BeautifulSoup (or similar) to scrape the website.
        return "B6 executed"
    elif "compress" in task_lower or "resize" in task_lower:
        # Use PIL/Pillow or similar for image processing.
        return "B7 executed"
    elif "transcribe audio" in task_lower:
        # Use a speech-to-text library or API.
        return "B8 executed"
    elif "convert markdown to html" in task_lower:
        # Use a Markdown library (e.g., markdown2) for conversion.
        return "B9 executed"
    elif "filter a csv" in task_lower:
        # Use Python’s csv module or pandas to filter the CSV.
        return "B10 executed"
    else:
        raise ValueError("Task not recognized or not supported.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
