import json
import time
import os
import anthropic
import multiprocessing
from anthropic import Tokenizer
from tqdm import tqdm

# Claude API setup
client = anthropic.Anthropic(api_key="your-anthropic-api-key")
tokenizer = Tokenizer()

# Token limit (Claude-compatible)
MAX_TOTAL_TOKENS = 40_000_000

# Category weights for proportional generation
CATEGORY_WEIGHTS = {
    "Common Sense": 3,
    "World Understanding": 3,
    "Math": 2,
    "Science": 2,
    "Medicine": 2,
    "Coding": 2,
    "Genetics": 1,
    "Technology": 1
}

# Build the OpenLLM-QA style prompt
def build_prompt(category: str) -> str:
    return (
        f"You are contributing to the dataset 'OpenLLM-QA'.\n"
        f"Please generate a single, clear, helpful question about the topic: **{category}**.\n"
        f"Then write a detailed, friendly and informative answer in natural, fluent English.\n\n"
        f"Format:\nQ: <question>\nA: <answer>"
    )

# Claude token counting
def count_claude_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Choose next category proportionally
def choose_category(local_counts: dict) -> str:
    current_ratios = {
        cat: local_counts[cat] / CATEGORY_WEIGHTS[cat] if CATEGORY_WEIGHTS[cat] > 0 else 1
        for cat in CATEGORY_WEIGHTS
    }
    return min(current_ratios, key=current_ratios.get)

# Worker process logic
def worker_main(worker_id: int, output_path: str, total_tokens, lock):
    local_counts = {cat: 0 for cat in CATEGORY_WEIGHTS}

    while True:
        with lock:
            if total_tokens.value >= MAX_TOTAL_TOKENS:
                print(f"[Worker {worker_id}] Max token limit reached. Exiting.")
                break

        category = choose_category(local_counts)
        prompt = build_prompt(category)

        try:
            response = client.messages.create(
                model="claude-3.5-sonnet-20240620",  # <- CLAUDE 3.5!
                max_tokens=1024,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text
            if "Q:" in content and "A:" in content:
                question = content.split("Q:")[1].split("A:")[0].strip()
                answer = content.split("A:")[1].strip()
                tokens = count_claude_tokens(question) + count_claude_tokens(answer)

                qa = {
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "tokens": tokens
                }

                with lock:
                    if total_tokens.value + tokens > MAX_TOTAL_TOKENS:
                        break
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    total_tokens.value += tokens
                    local_counts[category] += 1

        except anthropic.RateLimitError:
            print(f"[Worker {worker_id}] Rate limit hit. Sleeping 10s...")
            time.sleep(10)
        except anthropic.APIStatusError as e:
            print(f"[Worker {worker_id}] API error: {e}. Retrying...")
            time.sleep(5)
        except Exception as e:
            print(f"[Worker {worker_id}] Unknown error: {e}")
            time.sleep(3)

# Main function: start all processes
def generate_parallel_dataset(output_file="openllm_qa_claude35.jsonl", num_workers=5):
    manager = multiprocessing.Manager()
    total_tokens = manager.Value("i", 0)
    lock = manager.Lock()

    if not os.path.exists(output_file):
        open(output_file, "w").close()

    print(f"Starting generation with {num_workers} workers...")
    workers = []

    try:
        for i in range(num_workers):
            p = multiprocessing.Process(target=worker_main, args=(i, output_file, total_tokens, lock))
            p.start()
            workers.append(p)

        for p in workers:
            p.join()

    except KeyboardInterrupt:
        print("\n⛔ Interrupted. Waiting for workers to stop...")
        for p in workers:
            p.terminate()
        for p in workers:
            p.join()

    print(f"\n✅ Generation complete. Final token count: {total_tokens.value:,}")
    print(f"Dataset saved to: {output_file}")

if __name__ == "__main__":
    generate_parallel_dataset()
