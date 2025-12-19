"""
Master Pipeline Script - Run Complete Association Mining Workflow
"""
import subprocess
import sys
import time
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and report status"""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    start = time.time()
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True
    )
    
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"\n Completed in {elapsed:.2f}s")
        return True
    else:
        print(f"❌ Error running {script_name}:")
        print(result.stderr)
        return False

def main():
    print("="*80)
    print("MOVIE ASSOCIATION MINING PIPELINE")
    print("="*80)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    pipeline_start = time.time()
    
    # Step 1: Clean dataset
    if not run_script("src/preprocessing/clean_dataset.py", "Data Cleaning & Synonym Consolidation"):
        print("\n  Pipeline stopped due to error in data cleaning")
        return 1
    
    # Step 2: Format transactions
    if not run_script("src/preprocessing/format_dataset.py", "Transaction Formatting & Tag Selection"):
        print("\n Pipeline stopped due to error in formatting")
        return 1
    
    # Step 3: Run association mining (optional - via notebook)
    print("\n" + "="*80)
    print("STEP: Association Rule Mining")
    print("="*80)
    print("ℹ  To run association mining, open and execute:")
    print("   notebooks/assoc_rule_mining.ipynb")
    print("\nOr run programmatically:")
    print("   jupyter nbconvert --execute --to notebook notebooks/assoc_rule_mining.ipynb")
    
    total_elapsed = time.time() - pipeline_start
    
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f" Data cleaning: Complete")
    print(f" Transaction formatting: Complete")
    print(f" Ready for association mining")
    print(f"\nTotal time: {total_elapsed:.2f}s")
    print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check output files
    print("\n Output files generated:")
    outputs = [
        "datasets/cleaned/genome-scores-cleaned.csv",
        "datasets/cleaned/genome-tags-cleaned.csv",
        "datasets/formatted/transactions.csv",
    ]
    
    for output in outputs:
        path = Path(output)
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  ✓ {output} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {output} (not found)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
