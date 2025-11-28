import csv
import os
import subprocess

class MLLMEngine:
    def __init__(self, model_name="llama3"):
        self.model = model_name

    def ask(self, prompt):
        result = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt,
            text=True,
            capture_output=True,
            encoding='utf-8',
            errors='replace'
        )
        return result.stdout.strip()

    def describe_scene(self, summary):
        prompt = f"""
You are a helpful assistant that describes scenes based on detected object counts.

Here is the detected object summary:
{summary}

Please provide a vivid and engaging textual description of the scene.
"""
        return self.ask(prompt)


def save_descriptions_to_csv(image_descriptions, filepath):
    # image_descriptions: dict {image_filename: description}
    with open(filepath, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_filename", "description"])
        for img_name, desc in image_descriptions.items():
            writer.writerow([img_name, desc])


def main():
    image_descriptions = {}

    # Example list of images and their summaries (replace with your actual data)
    images_and_summaries = [
        ("img1.jpg", "12 cars, 6 traffic lights, 4 traffic signs, 2 motors, 1 rider"),
        ("img2.jpg", "5 persons, 2 trucks, 3 cars"),
        # add your images and summaries here
    ]

    llm = MLLMEngine(model_name="llama3")

    for img_name, summary in images_and_summaries:
        description = llm.describe_scene(summary)
        image_descriptions[img_name] = description
        print(f"Saved description for {img_name}")

    save_descriptions_to_csv(image_descriptions, "D:/YOLO_MLLM_Project/image_descriptions.csv")
                        

