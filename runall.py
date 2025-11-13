import subprocess
import sys

steps = [
    "python src/preprocess.py",
    "python src/train_model.py",
    "python src/evaluate_model.py",
    "python src/inference.py",
    "python src/monitor.py",
]

print("=" * 60)
print("Starting end-to-end MLOps pipeline")
print("=" * 60)

for step in steps:
    print(f"\nRunning: {step}")
    result = subprocess.run(step, shell=True)
    if result.returncode != 0:
        print(f"Step failed: {step}")
        sys.exit(1)
    else:
        print(f"Completed: {step}")

print("\nLaunching Streamlit dashboard...")
subprocess.Popen("streamlit run src/dashboard_py.py", shell=True)

print("\nPipeline execution finished. Dashboard is running.")
