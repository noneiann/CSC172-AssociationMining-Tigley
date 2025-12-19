import pandas as pd
import ast
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from visualization.visualizations import create_all_visualizations

def main():
    print("Loading data...")
    
    # Load transactions
    df_transactions = pd.read_csv('datasets/formatted/transactions.csv')
    df_transactions['tags'] = df_transactions['tags'].apply(ast.literal_eval)
    
    print(f"Loaded {len(df_transactions)} transactions with {df_transactions['tags'].apply(len).sum()} total tags")
    
    # Check if rules exist (from notebook run)
    try:
        # Try to load pre-generated rules
        import pickle
        with open('datasets/rules.pkl', 'rb') as f:
            rules = pickle.load(f)
        print(f"Loaded {len(rules)} association rules")
    except FileNotFoundError:
        print("\n  No rules found. Please run the association mining notebook first.")
        print("   Open notebooks/assoc_rule_mining.ipynb and execute cells 1-3 to generate rules.")
        return
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    all_figs = create_all_visualizations(
        rules, 
        df_transactions,
        save_dir='output/visualizations',
        network_top_n=200,
        heatmap_top_n=50
    )
    
    print(f"\n Complete! Generated {len(all_figs)} visualizations")
    print(" Files saved in: output/visualizations/")
    
    # Show what was created
    import os
    if os.path.exists('output/visualizations'):
        files = [f for f in os.listdir('output/visualizations') if f.endswith('.png')]
        print(f"\nGenerated files ({len(files)}):")
        for f in sorted(files):
            path = os.path.join('output/visualizations', f)
            size_kb = os.path.getsize(path) / 1024
            print(f"  ✓ {f} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
