import os
import random
import pandas as pd
import torch
import argparse
from transformers import GPT2Config

GPT2_CACHE_DIR = "./models/gpt2-cache"
DATA_FILE      = "./dataset/retail_inventory.csv"
CHECKPOINT     = "./checkpoints/retail/checkpoint.pth"

def pick_from_list(prompt, options):
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        choice = input(f"Enter choice [1-{len(options)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid selection, try again.")

def predict_units_sold():
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    H = {"1 Day":1, "1 Week":7, "2 Weeks":14, "3 Weeks":21}
    horizon_label = pick_from_list("Select forecast horizon:", list(H.keys()))
    days_ahead    = H[horizon_label]
    region   = pick_from_list("Select REGION:",   sorted(df["Region"].unique()))
    category = pick_from_list("Select CATEGORY:", sorted(df["Category"].unique()))
    slice_all  = df[(df["Region"]==region) & (df["Category"]==category)]
    product_id = random.choice(slice_all["Product ID"].unique().tolist())
    store_id   = slice_all[slice_all["Product ID"]==product_id]["Store ID"].iloc[0]
    slice_df   = slice_all[slice_all["Product ID"]==product_id]

    current_inventory = slice_df.sort_values("Date")["Inventory Level"].iloc[-1]

    discount      = float(input("Enter DISCOUNT (%): ").strip())
    holiday       = input("Holiday/Promotion? (Y/N): ").strip().lower().startswith("y")
    comp_price    = float(input("Enter COMPETITOR PRICING: ").strip())
    weather       = pick_from_list("Select WEATHER:",    sorted(df["Weather Condition"].unique()))
    seasonality   = pick_from_list("Select SEASONALITY:", sorted(df["Seasonality"].unique()))
    desired_price = float(input("Enter desired price: ").strip())

#INJECT user inputs into the slice 
    slice_df = slice_df.copy()
    slice_df["Discount"]           = discount
    slice_df["Holiday/Promotion"]  = "Yes" if holiday else "No"
    slice_df["Competitor Pricing"] = comp_price
    slice_df["Weather Condition"]  = weather
    slice_df["Seasonality"]        = seasonality

    required_len = 96 + pred_len
    while len(slice_df) < required_len:
        slice_df = slice_df.append(slice_df.iloc[-1], ignore_index=True)

    slice_df.to_csv("slice.csv", index=False)

    ckpt       = torch.load(CHECKPOINT, map_location="cpu")
    pred_len   = ckpt["output_projection.linear.bias"].shape[0]
    llm_cfg    = GPT2Config.from_pretrained(GPT2_CACHE_DIR)
    llm_layers = llm_cfg.num_hidden_layers

    args = argparse.Namespace(
        model           = "TimeLLM",
        task_name       = "long_term_forecast",
        data            = "retail",
        root_path       = ".",
        data_path       = "slice.csv",
        features        = "M",
        target          = "Units Sold",
        freq            = "d",
        percent         = 100,
        embed           = "timeF",
        seq_len         = 96,
        label_len       = 48,
        pred_len        = pred_len,
        batch_size      = 4,
        eval_batch_size = 4,
        learning_rate   = 1e-5,
        train_epochs    = 0,
        patience        = 0,
        num_workers     = 0,
        llm_model       = GPT2_CACHE_DIR,
        llm_dim         = 768,
        llm_layers      = llm_layers,
        n_heads         = 8,
        dropout         = 0.1,
        use_amp         = False,
        d_model         = 512,
        d_ff            = 512,
    )


    from data_provider.data_factory import data_provider
    ds, loader = data_provider(args, flag="test")
    args.enc_in = ds.enc_in

    from models.TimeLLM import Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Model(args).to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    if not hasattr(model, "top_k"):
        model.top_k = 5

    seq_x, seq_y, stm_x, stm_y = next(iter(loader))
    seq_x, seq_y, stm_x, stm_y = [t.to(device).float() for t in (seq_x, seq_y, stm_x, stm_y)]
    zeros   = torch.zeros_like(seq_y[:, -args.pred_len:, :]).to(device)
    dec_inp = torch.cat([seq_y[:, :args.label_len, :], zeros], dim=1)

    with torch.no_grad():
        out = model(seq_x, stm_x, dec_inp, stm_y)

    units_seq = out[0, :, 0].cpu().numpy()
    idx       = min(days_ahead-1, args.pred_len-1)
    raw_pred  = round(float(units_seq[idx]), 2)
    final_pred= round(abs(raw_pred) * 100, 2)

    return final_pred

if __name__ == "__main__":
    _ = predict_units_sold()
