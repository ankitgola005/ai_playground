import requests
import json

url = "http://127.0.0.1:8000/generate_stream"

while True:
    # Read user input
    user_input = input("Enter a prompt (or 'exit' to quit): ").strip()
    if user_input.lower() == "exit":
        break
    if not user_input:
        print("Please enter something!")
        continue

    data = {"prompts": [user_input], "max_tokens": 100, "use_cache": True}

    buffer = ""
    done = False

    print("\nGenerating...\n")

    with requests.post(url, json=data, stream=True) as response:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    line = line[len("data: ") :]
                obj = json.loads(line)

                if "token" in obj:
                    buffer += obj["token"]
                    print(f"\r{buffer}", end="")
                elif obj.get("done"):
                    done = True
                    print("\nGeneration done.\n")
                    break

    print(f"Final output: {buffer}\n")
