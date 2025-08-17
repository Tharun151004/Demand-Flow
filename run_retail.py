import os, subprocess


def run_retail_forecasting():
    ds_path = "./dataset/retail_inventory.csv"
    if not os.path.exists(ds_path):
        print(f"Dataset not found at {ds_path}")
        return

    import pandas as pd
    df = pd.read_csv(ds_path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())


    target_candidates = ['Units Sold']
    
    tgt = None
    for candidate in target_candidates:
        if candidate in df.columns:
            tgt = candidate
            break

    if tgt is None:
        lower_cols = [c.lower() for c in df.columns]
        for candidate in ['units sold', 'inventory level', 'sales', 'demand', 'quantity']:
            for i, col_lower in enumerate(lower_cols):
                if candidate in col_lower:
                    tgt = df.columns[i]
                    break
            if tgt:
                break
    

    if tgt is None:
        print("Available columns:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        
        choice = input("Enter the target column name or number: ").strip()
        

        try:
            idx = int(choice)
            if 0 <= idx < len(df.columns):
                tgt = df.columns[idx]
            else:
                print(f"Invalid column number. Must be between 0 and {len(df.columns)-1}")
                return
        except ValueError:

            if choice in df.columns:
                tgt = choice
            else:
                print(f"Column '{choice}' not found in data")
                return
    
    print("Using target:", tgt)

    seq_len = 96  
    patch_len = 16 
    stride = 8 
    pred_len = 24  
    
    # Calculate patch_nums for verification
    patch_nums = int((seq_len - patch_len) / stride + 1)
    print(f"Calculated patches: {patch_nums} (seq_len={seq_len}, patch_len={patch_len}, stride={stride})")
    
    cmd = [
        "python", "run_main.py",
        "--model", "TimeLLM",
        "--data", "retail",
        "--root_path", "./dataset",
        "--data_path", "retail_inventory.csv",
        "--features", "S",
        "--target", tgt,
        "--seq_len", "96",
        "--label_len", "48",
        "--pred_len", "24",
        "--batch_size", "4",            
        "--learning_rate", "1e-5",       
        "--train_epochs", "20",
        "--patience", "5",
        "--llm_model", "GPT2",
        "--llm_dim", "768",
        "--patch_len", "16",
        "--stride", "8",
        "--d_model", "512",
        "--d_ff", "512",          
        "--use_amp"
    ]

    print("RUNNING:", " ".join(cmd))
    subprocess.run(cmd)


def run_retail_forecasting_alternative():
    """Alternative approach with DLinear model which is more stable"""
    ds_path = "./dataset/retail_inventory.csv"
    if not os.path.exists(ds_path):
        print(f"Dataset not found at {ds_path}")
        return

    import pandas as pd
    df = pd.read_csv(ds_path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())


    target_candidates = ['Units Sold']
    
    tgt = None
    for candidate in target_candidates:
        if candidate in df.columns:
            tgt = candidate
            break
    
    if tgt is None:
        print("Available columns:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        
        choice = input("Enter the target column name or number: ").strip()
        
        try:
            idx = int(choice)
            if 0 <= idx < len(df.columns):
                tgt = df.columns[idx]
            else:
                print(f"Invalid column number. Must be between 0 and {len(df.columns)-1}")
                return
        except ValueError:
            if choice in df.columns:
                tgt = choice
            else:
                print(f"Column '{choice}' not found in data")
                return
    
    print("Using target:", tgt)
    print("Running with DLinear model (more stable alternative)")

    cmd = [
        "python", "run_main.py",
        "--model", "DLinear",
        "--data", "retail",
        "--root_path", "./dataset",
        "--data_path", "retail_inventory.csv",
        "--features", "S",
        "--target", tgt,
        "--seq_len", "96", 
        "--label_len", "48", 
        "--pred_len", "24",
        "--batch_size", "8", 
        "--learning_rate", "1e-3",
        "--train_epochs", "20", 
        "--patience", "5",
        "--moving_avg", "25", 
        "--use_amp"
    ]
    print("RUNNING:", " ".join(cmd))
    subprocess.run(cmd)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--dlinear":
        print("Running with DLinear model...")
        run_retail_forecasting_alternative()
    elif len(sys.argv) > 1 and sys.argv[1] == "--timellm":
        print("Running with TimeLLM model...")
        print("If you encounter issues, try: python run_retail.py --dlinear")
        run_retail_forecasting()
    else:
        print("Usage:")
        print("  python run_retail.py --timellm   # Run with TimeLLM model")
        print("  python run_retail.py --dlinear   # Run with DLinear model (more stable)")
        print("\nDefaulting to TimeLLM...")
        run_retail_forecasting()
