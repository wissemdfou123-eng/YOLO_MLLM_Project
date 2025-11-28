
def create_summary(detections):
    summary_lines = []
    for det in detections:
        class_counts = {}
        for c in det["classes"]:
            class_name = det["names"].get(int(c), "unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Format: "3 persons, 2 cars, 1 bike"
        parts = [f"{count} {cls}" + ("s" if count > 1 else "") for cls, count in class_counts.items()]
        summary_lines.append(", ".join(parts))
    return "\n".join(summary_lines)
