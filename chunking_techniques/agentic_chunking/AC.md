# Agentic Chunking — Code Explanation

## What is Agentic Chunking?

> Instead of splitting text by fixed size or punctuation, an **LLM acts as an agent** and intelligently decides where one topic ends and another begins.

---

## Full Working Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW INPUT TEXT                           │
│   "Mars rovers... Jezero Crater... ocean midnight zone..."      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               STEP 1 : get_propositions()                       │
│                                                                 │
│  LLM breaks text into ATOMIC FACTS (smallest standalone units)  │
│                                                                 │
│  Input  → Big paragraph                                         │
│  Output → Bulleted list of facts                                │
│                                                                 │
│  • Perseverance rover is on Mars                                │
│  • It searches for ancient microbial life                       │
│  • Jezero Crater is the search location                         │
│  • Scientists analyze soil samples                              │
│  • Midnight zone of ocean is largely a mystery                  │
│  • Giant squids live in crushing pressures                      │
│  • Bioluminescent fish thrive without sunlight                  │
│  • ROVs document alien-like species          ...etc             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          │  propositions = [p1, p2, p3, p4, p5...]
                          ▼
┌────────────────────────────────────────────────────────────────┐
│               STEP 2 : agentic_chunker()                       │
│                                                                │
│  current_chunk = propositions[0]   ← seed the first chunk      │
│                                                                │
│  Loop through remaining propositions one by one:               │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  for each next_proposition:                            │    │
│  │                                                        │    │
│  │   current_chunk  ──┐                                   │    │
│  │                    ▼                                   │    │
│  │          should_continue_chunk()   ← AGENT DECISION    │    │
│  │                    │                                   │    │
│  │          ┌─────────┴──────────┐                        │    │
│  │         YES                  NO                        │    │
│  │          │                   │                         │    │
│  │          ▼                   ▼                         │    │
│  │   Append to           Save current_chunk               │    │
│  │   current_chunk       to chunks[]                      │    │
│  │                       Start NEW current_chunk          │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                │
│  After loop ends → append last current_chunk to chunks[]       │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FINAL OUTPUT                                │
│                                                                 │
│   Chunk 1 → All Mars / Space related propositions               │
│   Chunk 2 → All Ocean / Marine related propositions             │
│   Chunk 3 → Extreme environments comparison (shared context)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Visual Flow of the Agent Decision Loop

```
propositions = [p1,  p2,  p3,  p4,  p5,  p6,  p7,  p8]
                │    │    │    │    │    │    │    │
                ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
Step:          seed  YES  YES  YES  NO   YES  YES  YES
                │                   │              │
                │                   │              │
             current              flush          flush
             _chunk               chunk1         chunk2
             builds               start          (last one
             up...                chunk2         appended
                                                 after loop)


chunks[] = [
   chunk1: "p1 + p2 + p3 + p4"   ← Mars topic
   chunk2: "p5 + p6 + p7 + p8"   ← Ocean topic
]
```

---

## Function-by-Function Breakdown

### `get_propositions(text)`

```
PURPOSE: Convert messy paragraph → clean list of atomic facts

INPUT  : "Mars rovers search Jezero Crater for ancient life..."
         (one big blob of text)

PROCESS: Sends to LLM with prompt:
         "Break this into standalone propositions as bullet list"

OUTPUT : [
           "Perseverance rover is in Jezero Crater",
           "Scientists search for ancient microbial life",
           "Soil samples are analyzed for liquid water signs",
           ...
         ]

WHY ATOMIC FACTS?
  → Each fact is independent
  → Makes agent decisions more accurate
  → Avoids splitting mid-sentence later
```

---

### `should_continue_chunk(current_chunk, next_proposition)`

```
PURPOSE: The core AGENT BRAIN — decides chunk boundaries

INPUT  : current_chunk    = "Perseverance searches Jezero Crater.
                             Soil samples analyzed for water."
         next_proposition = "Giant squids live in ocean midnight zone."

PROCESS: Sends BOTH to LLM with prompt:
         "Does next proposition continue the same topic? YES or NO"

OUTPUT : True  (if "YES" found in response)
         False (if "NO"  found in response)

EXAMPLE DECISIONS:
  ┌────────────────────────────────┬──────────┬─────────────┐
  │ current_chunk topic            │ next_prop│  Decision   │
  ├────────────────────────────────┼──────────┼─────────────┤
  │ Mars / Rovers / Jezero Crater  │ Mars soil│  YES ✅     │
  │ Mars / Rovers / Jezero Crater  │ Ocean    │  NO  ❌     │
  │ Ocean / Deep sea creatures     │ ROVs     │  YES ✅     │
  │ Ocean / Deep sea creatures     │ Sunlight │  YES ✅     │
  └────────────────────────────────┴──────────┴─────────────┘
```

---

### `agentic_chunker(text)`

```
PURPOSE: Orchestrator — ties everything together

PSEUDOCODE:
─────────────────────────────────────────────
propositions = get_propositions(text)
chunks       = []
current_chunk = propositions[0]          ← start with first fact

for p in propositions[1:]:
    if should_continue_chunk(current_chunk, p):   ← ASK AGENT
        current_chunk = current_chunk + " " + p   ← GROW chunk
    else:
        chunks.append(current_chunk)              ← SAVE chunk
        current_chunk = p                         ← RESET chunk

chunks.append(current_chunk)                      ← SAVE last chunk
return chunks
─────────────────────────────────────────────
```

---

## Data Flow Summary

```
RAW TEXT
   │
   │  get_propositions()
   │  ┌─────────────────────────────────────────┐
   │  │ LLM Call #1                             │
   │  │ Prompt: "Break into atomic facts"       │
   │  │ Returns: bulleted list                  │
   │  └─────────────────────────────────────────┘
   ▼
LIST OF PROPOSITIONS  [p1, p2, p3 ... pN]
   │
   │  agentic_chunker() loops over each proposition
   │
   │  For each pair (current_chunk, next_prop):
   │  ┌─────────────────────────────────────────┐
   │  │ LLM Call #2, #3, #4 ... #N              │
   │  │ Prompt: "Same topic? YES or NO"         │
   │  │ Returns: YES → merge | NO → split       │
   │  └─────────────────────────────────────────┘
   ▼
SEMANTICALLY GROUPED CHUNKS  [chunk1, chunk2 ...]
   │
   ▼
PRINTED OUTPUT per chunk
```

---

## Why This Approach is Powerful

| Traditional Chunking | Agentic Chunking |
|---|---|
| Split by fixed character count | Split by **semantic meaning** |
| Splits mid-sentence/mid-idea | Always keeps ideas **intact** |
| No understanding of content | LLM **understands** the content |
| Fast, no LLM calls | Slower but **intelligent** |
| Same result every time | Adapts to **any topic** |

---

## LLM Call Count

```
Total LLM Calls = 1 (get_propositions) + (N-1) (should_continue_chunk)

Where N = number of propositions extracted

Example:
  Text has 10 propositions
  → 1 + 9 = 10 total LLM calls
```

---
---

```
   Document Text
      │
      ▼
   split_into_sentences()          ← simple regex splitter
      │
      ▼
   For each sentence:
      │
      ├──► agent_decide_chunk()       ← LLM decides: existing or new?
      │         │
      │    existing ──► chunk.sentences.append()
      │         │       agent_update_summary()   ← LLM refreshes title+summary
      │         │
      │    new   ──► Chunk()
      │               agent_create_chunk_title_summary()  ← LLM titles it
      │
      ▼
   Final chunks list
```

---

# Agentic Chunking in Python

Agentic chunking is a technique where an LLM **decides** how to chunk text by acting as an agent — it reads each sentence and decides whether it belongs to an existing chunk or starts a new one.

Here's a full implementation from scratch:

````python
import json
import uuid
import os
from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"))
MODEL = "gpt-4o-mini"  # cheap & fast for agentic decisions


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

class Chunk:
    """Represents a single semantic chunk of text."""

    def __init__(self, chunk_id: str = None):
        self.chunk_id: str = chunk_id or str(uuid.uuid4())[:8]
        self.title: str = ""
        self.summary: str = ""
        self.sentences: list[str] = []

    def get_text(self) -> str:
        return " ".join(self.sentences)

    def to_dict(self) -> dict:
        return {
            "chunk_id":  self.chunk_id,
            "title":     self.title,
            "summary":   self.summary,
            "sentences": self.sentences,
            "full_text": self.get_text()
        }

    def __repr__(self):
        return (f"Chunk(id={self.chunk_id!r}, title={self.title!r}, "
                f"sentences={len(self.sentences)})")


# ─────────────────────────────────────────────
# LLM HELPERS
# ─────────────────────────────────────────────

def _chat(system: str, user: str, temperature: float = 0.0) -> str:
    """Minimal wrapper around OpenAI chat completion."""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
    )
    return resp.choices[0].message.content.strip()


def _json_chat(system: str, user: str) -> dict:
    """Like _chat but forces JSON output."""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
    )
    return json.loads(resp.choices[0].message.content)


# ─────────────────────────────────────────────
# AGENT DECISION FUNCTIONS
# ─────────────────────────────────────────────

def agent_decide_chunk(
    sentence: str,
    existing_chunks: list[Chunk]
) -> tuple[str, str]:
    """
    Agent decides:
      - Should this sentence go into an existing chunk?  → returns ('existing', chunk_id)
      - Should we create a new chunk?                    → returns ('new', '')

    The agent only sees chunk titles + summaries (not full text) to stay efficient.
    """
    if not existing_chunks:
        return ("new", "")

    # Build a compact summary of existing chunks for the agent
    chunks_info = "\n".join(
        f"  ID={c.chunk_id} | Title: {c.title} | Summary: {c.summary}"
        for c in existing_chunks
    )

    system = (
        "You are a semantic chunking agent. "
        "Your job is to decide where a new sentence belongs.\n"
        "Rules:\n"
        "1. If the sentence is closely related to an existing chunk's topic, assign it there.\n"
        "2. If the sentence introduces a clearly NEW topic, say 'new'.\n"
        "3. Prefer assigning to existing chunks unless the topic is genuinely different.\n"
        "Respond in JSON: {\"decision\": \"existing\" or \"new\", \"chunk_id\": \"<id or empty>\"}"
    )

    user = (
        f"New sentence:\n\"{sentence}\"\n\n"
        f"Existing chunks:\n{chunks_info}\n\n"
        "Where does this sentence belong?"
    )

    result = _json_chat(system, user)
    decision  = result.get("decision", "new")
    chunk_id  = result.get("chunk_id", "")
    return (decision, chunk_id)


def agent_update_summary(chunk: Chunk, new_sentence: str) -> tuple[str, str]:
    """
    After adding a sentence, the agent refreshes the chunk's title and summary.
    This keeps the summary current so future decisions are accurate.
    """
    system = (
        "You are a summarization agent. "
        "Given the current chunk content and a newly added sentence, "
        "produce an updated short title (5 words max) and a one-sentence summary.\n"
        "Respond in JSON: {\"title\": \"...\", \"summary\": \"...\"}"
    )

    user = (
        f"Current chunk text:\n\"{chunk.get_text()}\"\n\n"
        f"Newly added sentence:\n\"{new_sentence}\"\n\n"
        "Produce updated title and summary."
    )

    result = _json_chat(system, user)
    title   = result.get("title",   chunk.title)
    summary = result.get("summary", chunk.summary)
    return title, summary


def agent_create_chunk_title_summary(sentence: str) -> tuple[str, str]:
    """
    When a new chunk is created, generate its initial title and summary.
    """
    system = (
        "You are a summarization agent. "
        "Given a sentence that starts a new topic chunk, "
        "produce a short title (5 words max) and a one-sentence summary.\n"
        "Respond in JSON: {\"title\": \"...\", \"summary\": \"...\"}"
    )

    user = f"Opening sentence of new chunk:\n\"{sentence}\""

    result = _json_chat(system, user)
    title   = result.get("title",   "Untitled")
    summary = result.get("summary", sentence[:80])
    return title, summary


# ─────────────────────────────────────────────
# SENTENCE SPLITTER  (no NLTK needed)
# ─────────────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """
    Simple regex-free sentence splitter.
    Splits on '.', '!', '?' followed by whitespace or end-of-string.
    Good enough for most English text.
    """
    import re
    # Split but keep the delimiter with the sentence
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


# ─────────────────────────────────────────────
# CORE AGENTIC CHUNKER
# ─────────────────────────────────────────────

class AgenticChunker:
    """
    Main class that runs the agentic chunking loop.

    How it works:
    ─────────────
    For each sentence in the document:
      1. Ask the DECISION agent: existing chunk or new chunk?
      2a. If existing → add sentence to that chunk, ask SUMMARY agent to update title/summary.
      2b. If new      → create a new chunk, ask SUMMARY agent for initial title/summary.

    The agent never sees full text of other chunks — only titles + summaries.
    This keeps token usage low while preserving semantic accuracy.
    """

    def __init__(self, verbose: bool = True):
        self.chunks: dict[str, Chunk] = {}   # chunk_id → Chunk
        self.verbose = verbose

    # ── private ──────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    # ── public ───────────────────────────────

    def add_sentence(self, sentence: str):
        """Process one sentence through the agentic pipeline."""

        self._log(f"\n{'─'*60}")
        self._log(f"📝 Sentence: {sentence[:80]}...")

        existing = list(self.chunks.values())
        decision, chunk_id = agent_decide_chunk(sentence, existing)

        if decision == "existing" and chunk_id in self.chunks:
            # ── Add to existing chunk ──────────────
            chunk = self.chunks[chunk_id]
            chunk.sentences.append(sentence)
            new_title, new_summary = agent_update_summary(chunk, sentence)
            chunk.title   = new_title
            chunk.summary = new_summary
            self._log(f"  ✅ Added to existing chunk [{chunk_id}] → '{chunk.title}'")

        else:
            # ── Create new chunk ───────────────────
            new_chunk = Chunk()
            new_chunk.sentences.append(sentence)
            title, summary = agent_create_chunk_title_summary(sentence)
            new_chunk.title   = title
            new_chunk.summary = summary
            self.chunks[new_chunk.chunk_id] = new_chunk
            self._log(f"  🆕 New chunk [{new_chunk.chunk_id}] → '{new_chunk.title}'")

    def chunk_document(self, text: str) -> list[dict]:
        """
        Full pipeline:
          1. Split text into sentences.
          2. Feed each sentence to the agentic loop.
          3. Return final chunks as list of dicts.
        """
        sentences = split_into_sentences(text)
        total = len(sentences)
        self._log(f"🚀 Starting Agentic Chunking | {total} sentences detected\n")

        for i, sentence in enumerate(sentences, 1):
            self._log(f"[{i}/{total}]")
            self.add_sentence(sentence)

        self._log(f"\n{'═'*60}")
        self._log(f"✅ Done! Created {len(self.chunks)} chunks.")
        return self.get_chunks()

    def get_chunks(self) -> list[dict]:
        """Return all chunks as a list of dicts."""
        return [c.to_dict() for c in self.chunks.values()]

    def print_chunks(self):
        """Pretty-print all chunks."""
        print(f"\n{'═'*60}")
        print(f"  FINAL CHUNKS ({len(self.chunks)} total)")
        print(f"{'═'*60}")
        for c in self.chunks.values():
            print(f"\n🔷 Chunk ID : {c.chunk_id}")
            print(f"   Title    : {c.title}")
            print(f"   Summary  : {c.summary}")
            print(f"   Sentences: {len(c.sentences)}")
            print(f"   Text     : {c.get_text()[:200]}...")
        print(f"{'═'*60}\n")


# ─────────────────────────────────────────────
# MAIN — Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":

    sample_text = """
    The Amazon rainforest is the world's largest tropical rainforest, covering over 5.5 million square kilometers.
    It is home to an estimated 10% of all species on Earth.
    The forest plays a crucial role in regulating the global climate by absorbing carbon dioxide.
    Deforestation has been a major concern in recent decades due to logging and agricultural expansion.
    Scientists warn that losing the Amazon could trigger irreversible climate tipping points.

    Photosynthesis is the process by which plants convert sunlight into chemical energy.
    Chlorophyll in plant cells absorbs light, primarily in the red and blue wavelengths.
    The process produces glucose and oxygen as byproducts from carbon dioxide and water.
    Without photosynthesis, most life on Earth could not exist.

    The Python programming language was created by Guido van Rossum in 1991.
    Python emphasizes code readability and simplicity, making it beginner-friendly.
    It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    Python has become one of the most popular languages for data science and machine learning.
    Libraries like NumPy, Pandas, and TensorFlow are widely used in the Python ecosystem.

    Climate change refers to long-term shifts in global temperatures and weather patterns.
    Human activities, particularly burning fossil fuels, have accelerated climate change since the industrial revolution.
    The Amazon deforestation directly contributes to higher carbon emissions globally.
    International agreements like the Paris Accord aim to limit global warming to 1.5 degrees Celsius.
    """

    # ── Run Agentic Chunking ──
    chunker = AgenticChunker(verbose=True)
    chunks  = chunker.chunk_document(sample_text)

    # ── Print Results ──
    chunker.print_chunks()

    # ── Save to JSON ──
    with open("chunks_output.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print("💾 Saved to chunks_output.json")
````

---

## How It Works — Step by Step

```
Document Text
     │
     ▼
split_into_sentences()          ← simple regex splitter
     │
     ▼
For each sentence:
     │
     ├──► agent_decide_chunk()       ← LLM decides: existing or new?
     │         │
     │    existing ──► chunk.sentences.append()
     │         │       agent_update_summary()   ← LLM refreshes title+summary
     │         │
     │    new   ──► Chunk()
     │               agent_create_chunk_title_summary()  ← LLM titles it
     │
     ▼
Final chunks list
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| Agent sees only **titles + summaries** | Keeps token cost low vs feeding full text |
| **Summary is updated** after each sentence added | Future decisions stay accurate |
| **UUID-based chunk IDs** | Easy to reference and return to |
| No LangChain | Pure OpenAI SDK + stdlib only |
| `response_format: json_object` | Reliable structured output from LLM |
