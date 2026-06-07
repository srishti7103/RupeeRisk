import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n==================================================")
    print(f"Executing: {script_name}")
    print(f"==================================================")
    
    # Determine the python executable inside the active environment
    python_bin = sys.executable if sys.executable else "python"
    
    # Run the script and stream the output
    result = subprocess.run([python_bin, script_name], capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR]: {script_name} failed with return code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    else:
        print(f"[SUCCESS]: {script_name} completed successfully.")

if __name__ == "__main__":
    print("Starting RupeeRisk Automatic Update Pipeline...")
    
    # Step 1: Collect latest data and build master features
    run_script("collect_data.py")
    
    # Step 2: Run forecasting backtest across all 9 models
    run_script("run_pipeline.py")
    
    # Step 3: Compute out-of-sample forecast signal for next week
    run_script("generate_next_week_signal.py")
    
    print("\n==================================================")
    print("RupeeRisk Platform updated successfully!")
    print("==================================================")
