import os
import subprocess
from itertools import combinations

# Configuration
DATA_DIR = "/home/exouser/pruning/induction_datasets/category_chains"
SCRIPT_PATH = "/home/exouser/pruning/intersection_experiments/run_sequential_intersection.py"
PRUNING_LEVEL = "heads"
EPOCHS_JOINT = 100
EPOCHS_FINE = 100

def get_datasets():
    # Fetch all jsonl files in the category chains directory
    datasets = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".jsonl"):
            datasets.append(os.path.join(DATA_DIR, f))
    return sorted(datasets)

def main():
    datasets = get_datasets()
    pairs = list(combinations(datasets, 2))
    
    print(f"Found {len(datasets)} datasets. Generating {len(pairs)} pairwise intersection tasks.")
    
    processes = []
    
    for i, (task_a, task_b) in enumerate(pairs, 1):
        name_a = os.path.basename(task_a).split('.')[0]
        name_b = os.path.basename(task_b).split('.')[0]
        print(f"Launching PAIR {i}/{len(pairs)}: {name_a.upper()} vs {name_b.upper()} in background...")

        
        cmd = [
            "/home/exouser/pruning/venv/bin/python",
            SCRIPT_PATH,
            "--task_a", task_a,
            "--task_b", task_b,
            "--epochs_joint", str(EPOCHS_JOINT),
            "--epochs_fine", str(EPOCHS_FINE),
            "--level", PRUNING_LEVEL
        ]
        
        # Create log files for each concurrent process to avoid output scrambling
        log_path = os.path.join("/home/exouser/pruning/intersection_experiments", f"log_{name_a}_vs_{name_b}.txt")
        log_file = open(log_path, "w")
        
        # Run the command concurrently
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, name_a, name_b, log_file))
        
    print(f"\nAll {len(processes)} processes launched concurrently! Waiting for them to finish...")
    
    # Wait for all processes
    for p, name_a, name_b, log_file in processes:
        p.wait()
        log_file.close()
        if p.returncode != 0:
            print(f"ERROR: {name_a} vs {name_b} failed (Check log_{name_a}_vs_{name_b}.txt)")
        else:
            print(f"SUCCESS: {name_a} vs {name_b} finished successfully.")
            
    print("\nAll batch pairwise intersections completed!")

if __name__ == "__main__":
    main()
