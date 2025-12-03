import subprocess
import time
import re
import pandas as pd

def run_model(script_path):
    print(f"Running {script_path}...")
    start_time = time.time()
    try:
        result = subprocess.run(
            ["uv", "run", script_path],
            capture_output=True,
            text=True,
            check=False 
        )
        end_time = time.time()
        duration = end_time - start_time
        
        output = result.stdout
        error = result.stderr
        
        # Parse accuracy
        accuracy = None
        
        # Check for different accuracy patterns
        if "Test accuracy:" in output:
            match = re.search(r"Test accuracy:\s*([0-9.]+)", output)
            if match:
                accuracy = float(match.group(1))
        elif "Accuracy" in output:
            # Look for the number following "Accuracy"
            lines = output.splitlines()
            for i, line in enumerate(lines):
                if "Accuracy" in line:
                    # Try next line
                    if i + 1 < len(lines):
                        try:
                            accuracy = float(lines[i+1].strip())
                            break
                        except ValueError:
                            pass
                    # Try same line
                    match = re.search(r"Accuracy\s*[:]?\s*([0-9.]+)", line)
                    if match:
                        accuracy = float(match.group(1))
                        break

        return {
            "script": script_path,
            "accuracy": accuracy,
            "time": duration,
            "output": output,
            "error": error,
            "return_code": result.returncode
        }
        
    except Exception as e:
        return {
            "script": script_path,
            "accuracy": None,
            "time": 0,
            "output": "",
            "error": str(e),
            "return_code": -1
        }

models = [
    {"path": "src/main_svm.py", "name": "SVM", "desc": "RBF Kernel, C=100, gamma=1.0"},
    {"path": "src/main.KNN.py", "name": "KNN", "desc": "K=1"},
    {"path": "src/main_torch.py", "name": "Neural Network", "desc": "MLP (27->8->9), ReLU, Adam"},
    {"path": "src/main.py", "name": "LightGBM", "desc": "Gradient Boosting, num_leaves=10, max_depth=2"}
]

results = []
for model in models:
    res = run_model(model["path"])
    res["name"] = model["name"]
    res["desc"] = model["desc"]
    results.append(res)

print("\n" + "="*80)
print(f"{'Model':<20} | {'Accuracy':<10} | {'Time (s)':<10} | {'Status'}")
print("-" * 80)

latex_rows = []

for res in results:
    status = "OK" if res["return_code"] == 0 and res["accuracy"] is not None else "FAIL"
    acc_str = f"{res['accuracy']:.4f}" if res['accuracy'] is not None else "N/A"
    print(f"{res['name']:<20} | {acc_str:<10} | {res['time']:<10.2f} | {status}")
    
    if status == "FAIL":
        print(f"Error output for {res['name']}:\n{res['error']}\n{res['output']}\n")

    # Prepare LaTeX row
    # Format: Model & Description & Accuracy & Time \\
    latex_rows.append(f"{res['name']} & {res['desc']} & {acc_str} & {res['time']:.2f} \\\\")

print("\n" + "="*80)
print("\nLaTeX Table Body:\n")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Comparison of different models on the VSSS color classification task.}")
print("\\label{tab:model_comparison}")
print("\\begin{tabular}{l l c c}")
print("\\toprule")
print("\\textbf{Model} & \\textbf{Description} & \\textbf{Accuracy} & \\textbf{Time (s)} \\\\")
print("\\midrule")
for row in latex_rows:
    print(row)
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
