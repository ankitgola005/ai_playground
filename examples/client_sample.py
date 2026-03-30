# client_streaming.py
import requests
import json

url = "http://127.0.0.1:8000/generate_stream"
data = {
    "prompts": ["To be or not to be", "O Romeo, Romeo!"],
    "max_tokens": 20,
    "use_cache": True,
}

# Initialize a buffer for each prompt
prompt_buffers = {i: "" for i in range(len(data["prompts"]))}
done_prompts = set()

with requests.post(url, json=data, stream=True) as response:
    for line in response.iter_lines(decode_unicode=True):
        if line:
            # Strip "data: " prefix if present
            if line.startswith("data: "):
                line = line[len("data: ") :]
            obj = json.loads(line)

            pid = obj["prompt_id"]
            if "token" in obj:
                # Append streamed token to the buffer
                prompt_buffers[pid] += obj["token"]
                print(f"\rPrompt {pid}: {prompt_buffers[pid]}", end="")
            elif obj.get("done"):
                done_prompts.add(pid)
                print(f"\nPrompt {pid} generation done.")

        # Stop if all prompts are done
        if len(done_prompts) == len(data["prompts"]):
            break

# Final outputs
for pid, text in prompt_buffers.items():
    print(f"\nFinal output for Prompt {pid}: {text}")
