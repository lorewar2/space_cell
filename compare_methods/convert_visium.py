import scanpy as sc
import pandas as pd

adata = sc.read_mtx("gene_expression.mtx").T
genes = pd.read_csv("genes.tsv", header=None, sep="\t")
barcodes = pd.read_csv("barcodes.tsv", header=None, sep="\t")

adata.var_names = genes.iloc[:, 1]
adata.obs_names = barcodes.iloc[:, 0]

df = adata.to_df().T
df.index.name = "Row"
df.to_csv("gene_cell_matrix.csv")