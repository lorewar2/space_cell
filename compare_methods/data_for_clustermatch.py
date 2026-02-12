import pandas as pd
import scanpy as sc
import numpy as np

SPATIAL_COUNT = "./data/cosmx_count.csv"
SPATIAL_ANNO = "./data/cosmx_anno.csv" # cellcharter cluster output
SPATIAL_GENE = "./data/cosmx_gene.csv"

SCRNA_COUNT = "./data/scrna_count.csv"
SCRNA_GENE = "./data/scrna_gene.csv"
SCRNA_ANNO = "./data/scrna_anno.csv"

def main():
    #scrna_data_maker()
    spatial_data_maker()
    return


def scrna_data_maker():
    # load files
    anno_df = pd.read_csv(SCRNA_ANNO, nrows=20_000)
    count_df = pd.read_csv(SCRNA_COUNT, header=None, nrows=20_000)
    gene_df = pd.read_csv(SCRNA_GENE)

    # 1. select 1000 cells per broadcelltype
    selected_cells = (
        anno_df
        .groupby("broad_celltypes", group_keys=False)
        .apply(lambda x: x.sample(n=min(1000, len(x)), random_state=42))
    )

    # get indices of selected cells
    selected_idx = selected_cells.index

    # subset count matrix (rows = cells)
    subset_counts = count_df.iloc[selected_idx]

    # transpose so rows = genes, columns = cells
    subset_counts = subset_counts.T

    # adjust column name below if gene symbol column has a different name
    if "gene_symbol" in gene_df.columns:
        subset_counts.index = gene_df["gene_symbol"]
    else:
        subset_counts.index = gene_df.iloc[:, 0]

    # set cell IDs as column names
    if "cell_id" in selected_cells.columns:
        cell_ids = selected_cells["cell_id"].values
    else:
    # if no explicit cell_id column, create from original index
        cell_ids = selected_idx.astype(str).values

    subset_counts.columns = cell_ids
    # save subset
    subset_counts.to_csv("scrna_expression.csv")

    # save annotation
    annotation_out = pd.DataFrame({
        "cell_id": cell_ids,
        "broad_celltypes": selected_cells["broad_celltypes"].values
    })

    annotation_out.to_csv("scrna_cell_annotation.csv", index=False)
    return 

def spatial_data_maker():
    # load files
    anno_df = pd.read_csv(SPATIAL_ANNO)
    count_df = pd.read_csv(SPATIAL_COUNT)
    gene_df = pd.read_csv(SPATIAL_GENE)

    # extract cell IDs (column 1) and clusters (column 4)
    cell_ids = anno_df.iloc[:, 0].astype(str).values
    clusters = anno_df.iloc[:, 3].values

    # extract gene symbols (skip first row which just says "gene")
    gene_symbols = gene_df.iloc[:, 0].values

    # ensure gene list length matches count rows
    print(len(gene_symbols))
    print(count_df.shape[0])
    if len(gene_symbols) != count_df.shape[0]:
        raise ValueError("Gene list length does not match count matrix rows.")

    # assign gene symbols as row names
    count_df.index = gene_symbols

    # assign cell IDs as column names
    count_df.columns = cell_ids

    # xave expression matrix
    count_df.to_csv("spatial_expression.csv")

    # xave annotation file
    annotation_out = pd.DataFrame({
        "cell_id": cell_ids,
        "cluster": clusters
    })

    annotation_out.to_csv("spatial_cell_annotation.csv", index=False)
    return

if __name__ == '__main__':
    main()