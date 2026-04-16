
import pandas as pd
from sklearn.metrics import adjusted_rand_score

def rand_index_compare():
    # load files (no header assumed; adjust if needed)
    df1 = pd.read_csv("./data/visium_manual.csv")
    df2 = pd.read_csv("./leiden_cere.csv")

    df1_sub = df1.iloc[:, [1, 2]].copy()
    df1_sub.columns = ["barcode", "cluster1"]

    print("done")
    df2_sub = df2.iloc[:, [0, 1]].copy()
    df2_sub.columns = ["barcode", "cluster2"]

    # merge on common barcodes
    merged = pd.merge(df1_sub, df2_sub, on="barcode", how="inner")

    if merged.empty:
        raise ValueError("No overlapping barcodes found.")

    # compute ARI
    ari = adjusted_rand_score(merged["cluster1"], merged["cluster2"])

    print(ari)
    return



if __name__ == "__main__":
    rand_index_compare()