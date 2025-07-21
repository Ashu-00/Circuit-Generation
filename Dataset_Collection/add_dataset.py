import json
from datasets import Dataset, load_dataset

HF_api = ""

def convert_json_to_dataset(json_file):
    """
    Converts a JSON file containing circuit data to a Hugging Face Dataset.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        Dataset: A Hugging Face Dataset.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)
    return dataset

def upload_dataset_to_hub(dataset, hub_name):
    """
    Uploads a Hugging Face Dataset to the Hugging Face Hub.

    Args:
        dataset (Dataset): The Hugging Face Dataset to upload.
        hub_name (str): The desired name of the dataset repository on the Hugging Face Hub.
    """
    dataset.push_to_hub(hub_name, token=HF_api)
    print(f"Dataset uploaded to the Hugging Face Hub: {hub_name}")

if __name__ == '__main__':
    json_file = 'circuit_analysis_prompts.json'  # Replace with the actual path to your JSON file
    hub_name = 'Ashed00/SPICE-Circuits'  # Replace with your desired Hugging Face Hub repository name

    dataset = convert_json_to_dataset(json_file)
    print(dataset)

    upload_dataset_to_hub(dataset, hub_name)
