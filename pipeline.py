import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Run_YOLO import YOLORunner
from MLLM.mllm_core import MLLMEngine
from utils import create_summary  # or wherever you put the summary function

def main():
    print("🔹 Running YOLO predictions...")
    yolo = YOLORunner(model_path="data/best.pt")
    detections = yolo.predict("data/images", save=False)
    
    summary = create_summary(detections)
    print("Detected Objects Summary:")
    print(summary)

    print("🔹 Sending summary to MLLM...")
    llm = MLLMEngine(model_name="llama3")
    description = llm.describe_scene({"summary": summary})

    print("\nMLLM Description of Scene:")
    print(description)

    # Optional: if you want to combine results with collaboration module
    collab = CollaborationModule()
    final_report = collab.combine({"summary": summary}, description)

    print("\n=========== FINAL REPORT ===========")
    print(final_report)

if __name__ == "__main__":
    main()
