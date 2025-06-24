import asyncio, os, json, time
from typing import Dict, Any
from pydantic import BaseModel
import chromadb # type: ignore
from chromadb.config import Settings # type: ignore
from openai import AsyncOpenAI # type: ignore

# ---------- config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")        # export before running
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
db = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet"))   # local folder ./chroma
collection = db.get_or_create_collection("memories")

# ---------- tool registry ----------
class EchoArgs(BaseModel):
    text: str

async def echo(text: str) -> str:
    await asyncio.sleep(0)        # yield to event loop
    return text

TOOLS: Dict[str, tuple[type[BaseModel], Any]] = {"echo": (EchoArgs, echo)}

# ---------- chat helper ----------
SYSTEM_PROMPT = """
You are Matt’s personal assistant. 
If you need to call a tool, return JSON like {"name": "...", "arguments": {...}}.
Valid tools:
- echo(text:str) → str  # repeats whatever you send
"""

async def chat(messages):
    """OpenAI chat with function-calling mode."""
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return resp.choices[0].message

# ---------- main loop ----------
async def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        user = input("\nYou: ").strip()
        if not user: 
            continue

        # embed and store new message
        emb = await client.embeddings.create(
            model="text-embedding-3-small", input=[user]
        )
        collection.add(
            ids=[f"{time.time_ns()}"], embeddings=[emb.data[0].embedding],
            documents=[user], metadatas=[{"role": "user"}]
        )

        # retrieve top-k memories
        hits = collection.query(
            query_embeddings=[emb.data[0].embedding], n_results=5, include=["documents"]
        )
        recalled = "\n".join(hits["documents"][0]) if hits["documents"] else ""

        # assemble prompt
        prompt_block = (
            f"**Most relevant memories**\n{recalled}\n\n"
            f"**Current message**\n{user}"
        )
        messages.append({"role": "user", "content": prompt_block})

        # ask model
        assistant = await chat(messages)
        try:
            payload = json.loads(assistant.content)        # tool call?
        except json.JSONDecodeError:
            print("Assistant:", assistant.content)
            messages.append({"role": "assistant", "content": assistant.content})
            continue

        name, args = payload.get("name"), payload.get("arguments", {})
        if name not in TOOLS:
            print("Assistant (invalid tool):", assistant.content)
            messages.append({"role": "assistant", "content": assistant.content})
            continue

        schema, fn = TOOLS[name]
        try:
            typed_args = schema(**args)
        except Exception as e:
            print("Arg error:", e)
            continue

        result = await fn(**typed_args.dict())             # run tool
        tool_reply = json.dumps({"tool": name, "result": result})
        print("Tool result:", result)

        # feed result back so model knows outcome
        messages.extend([
            {"role": "assistant", "content": assistant.content},
            {"role": "tool", "name": name, "content": tool_reply},
        ])

if __name__ == "__main__":
    asyncio.run(main())
