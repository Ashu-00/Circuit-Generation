import os
import json
from google import genai
from google.genai import types
from tqdm import tqdm
import time


# Configure the Gemini API (replace with your actual API key)
GOOGLE_API_KEY = ""  # Replace with your actual API key
client = genai.Client(api_key=GOOGLE_API_KEY)


def create_gemini_prompt(content):
    """
    Generates a more descriptive prompt using the Gemini API.
    """
    prompt = f"Based on the given SPICE Netlist, create a description of the circuit's functionality that can be used as a prompt to generate this circuit: \n {content} \n Answer with only the prompt and do not include any additional text or explanations."
    try:
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt
        )
        return response.text
    except genai.errors.ClientError as e:
        print(f"ClientError encountered: {e}")
        retry_seconds = 60  # Default fallback

        # Check if e.args exists and has at least one element
        if e.args and isinstance(e.args[0], dict):
            details = e.args[0].get("details", [])
            for detail in details:
                if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                    retry_delay = detail.get("retryDelay", "1s")
                    try:
                        retry_seconds = int(retry_delay[:-1]) + 1 # Extract number and add 1
                    except ValueError:
                        retry_seconds = 60
                        print(f"Warning: Could not parse retryDelay '{retry_delay}'. Using default {retry_seconds}s.")
                    break
        #else:
        #    print("Error argument is not a dictionary, or no arguments provided. Retrying with default delay.")

        print(f"Retrying in {retry_seconds} seconds...")
        time.sleep(retry_seconds)
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Error generating Gemini prompt: {e}")
        return f"Analyze the following KiCad circuit schematic with title: '{title}' and describe its functionality."

def create_prompts(directory):
    """
    Reads .cir files from a directory, extracts the title, and generates a prompt using Gemini.
    """
    prompts = []
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".cir"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    # Read whole file
                    content = f.read()
                    # first_line = f.readline().strip()
                    # if first_line.startswith(".title"):
                    #     title = first_line[7:].strip()  # Remove ".title" and any extra spaces
                    # else:
                    #     title = "No title found"
                    prompt = create_gemini_prompt(content)
                    prompts.append({"filename": filename, "prompt": prompt, "content": content})
                    # Write the prompt to a local text file
                    txt_output_path = "prompt.txt"
                    with open(txt_output_path, 'a') as txt_file:
                        txt_file.write(f"{filename}:{prompt}\n<end>")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return prompts

def write_prompts_to_json(prompts, output_file):
    """
    Writes the generated prompts to a JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=4)

if __name__ == "__main__":
    directory = "non_empty_cir_files"  # Directory containing .cir files
    output_file = "circuit_analysis_prompts.json"
    prompts = create_prompts(directory)
    write_prompts_to_json(prompts, output_file)
    print(f"Prompts generated and saved to {output_file}")
