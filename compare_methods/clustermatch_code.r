```{r}
lol <- function (reference_matrix, query_matrix, ref.res, que.res, 
  ref.norm = TRUE, que.norm = TRUE, dim.pca = 50, dim.cca = 30, 
  min.cluster.cells = 20, random_PCC = 0, distance_diff = 0, 
  distance_same = 0) 
{
  reference <- Seurat::CreateSeuratObject(counts = reference_matrix)
  if (ref.norm == TRUE) {
    reference <- Seurat::NormalizeData(reference)
  }
  reference <- Seurat::FindVariableFeatures(reference, selection.method = "vst", 
    nfeatures = 2000, verbose = FALSE)
  reference <- Seurat::ScaleData(reference, verbose = FALSE)
  reference <- Seurat::RunPCA(reference, features = Seurat::VariableFeatures(object = reference), 
    verbose = FALSE)
  reference <- Seurat::FindNeighbors(reference, dims = 1:dim.pca, 
    verbose = FALSE)
  reference <- Seurat::FindClusters(reference, resolution = ref.res)
  reference@meta.data$name <- "D1_"
  reference$seurat_clusters <- stringr::str_c(reference$name, 
    reference$seurat_clusters)
  reference@meta.data <- dplyr::select(reference@meta.data, 
    -c("name"))
  query <- Seurat::CreateSeuratObject(counts = query_matrix)
  if (que.norm == TRUE) {
    query <- Seurat::NormalizeData(query)
  }
  query <- Seurat::FindVariableFeatures(query, selection.method = "vst", 
    nfeatures = 2000, verbose = FALSE)
  query <- Seurat::ScaleData(query, verbose = FALSE)
  query <- Seurat::RunPCA(query, features = Seurat::VariableFeatures(object = query), 
    verbose = FALSE)
  query <- Seurat::FindNeighbors(query, dims = 1:dim.pca, 
    verbose = FALSE)
  query <- Seurat::FindClusters(query, resolution = que.res)
  query@meta.data$name <- "D2_"
  query$seurat_clusters <- stringr::str_c(query$name, query$seurat_clusters)
  query@meta.data <- dplyr::select(query@meta.data, -c("name"))
  CCA <- Seurat::RunCCA(object1 = reference, object2 = query, 
    num.cc = dim.cca)
  L2CCA <- Seurat::L2Dim(CCA, reduction = "cca")
  embedding <- L2CCA@reductions[["cca.l2"]]@cell.embeddings
  embedding <- data.frame(cbind(cells = rownames(embedding), 
    embedding))
  reference_cluster <- Seurat::SplitObject(reference, split.by = "seurat_clusters")
  reference_means <- data.frame()
  reference_cell_cluster_embedding <- list()
  cat("START\n")
  for (i in 1:length(reference_cluster)) {
    cluster_name <- names(reference_cluster)[i]
    cluster_cell <- data.frame(cells = rownames(reference_cluster[[i]]@meta.data))
    cluster_embedding <- dplyr::left_join(cluster_cell, 
      embedding, by = "cells")
    rownames(cluster_embedding) <- cluster_embedding[, 1]
    cluster_embedding <- cluster_embedding[, -1]
    reference_cell_cluster_embedding[[i]] <- as.data.frame(lapply(cluster_embedding, 
      as.numeric))
    rownames(reference_cell_cluster_embedding[[i]]) <- cluster_cell$cells
    reference_cell_cluster_embedding[[i]] <- t(reference_cell_cluster_embedding[[i]])
    names(reference_cell_cluster_embedding)[i] <- cluster_name
    cluster_embedding <- as.data.frame(lapply(cluster_embedding, 
      as.numeric))
    cluster_vec <- colMeans(cluster_embedding)
    cluster_vec <- data.frame(cluster_vec)
    colnames(cluster_vec)[1] <- cluster_name
    if (i == 1) {
      reference_means <- cluster_vec
    }
    else reference_means <- cbind(reference_means, cluster_vec)
  }
  query_cluster <- Seurat::SplitObject(query, split.by = "seurat_clusters")
  query_means <- data.frame()
  query_cell_cluster_embedding <- list()
  cat("START1\n")
  for (i in 1:length(query_cluster)) {
    cluster_name <- names(query_cluster)[i]
    cluster_cell <- data.frame(cells = rownames(query_cluster[[i]]@meta.data))
    cluster_embedding <- dplyr::left_join(cluster_cell, 
      embedding, by = "cells")
    rownames(cluster_embedding) <- cluster_embedding[, 1]
    cluster_embedding <- cluster_embedding[, -1]
    query_cell_cluster_embedding[[i]] <- as.data.frame(lapply(cluster_embedding, 
      as.numeric))
    rownames(query_cell_cluster_embedding[[i]]) <- cluster_cell$cells
    query_cell_cluster_embedding[[i]] <- t(query_cell_cluster_embedding[[i]])
    names(query_cell_cluster_embedding)[i] <- cluster_name
    cluster_embedding <- as.data.frame(lapply(cluster_embedding, 
      as.numeric))
    cluster_vec <- colMeans(cluster_embedding)
    cluster_vec <- data.frame(cluster_vec)
    colnames(cluster_vec)[1] <- cluster_name
    if (i == 1) {
      query_means <- cluster_vec
    }
    else query_means <- cbind(query_means, cluster_vec)
  }
  cat("START2\n")
  Seurat::Idents(object = reference) <- reference@meta.data$seurat_clusters
  reference_marker <- Seurat::FindAllMarkers(reference, only.pos = TRUE, 
    min.pct = 0.25, logfc.threshold = 0.25)
  reference_marker_cluster <- split(reference_marker, reference_marker$cluster)
  Seurat::Idents(object = query) <- query@meta.data$seurat_clusters
  query_marker <- Seurat::FindAllMarkers(query, only.pos = TRUE, 
    min.pct = 0.25, logfc.threshold = 0.25)
  query_marker_cluster <- split(query_marker, query_marker$cluster)
  #CCA_PCC <- cor(query_means, reference_means)
  CCA_PCC <- cor(as.matrix(query_means), as.matrix(reference_means))
  cat("START3\n")
  CCA_PCC[CCA_PCC < 0] <- 0
  jaccard <- function(a, b) {
    intersection = length(intersect(a, b))
    union = length(a) + length(b) - intersection
    return(intersection/union)
  }
  marker_jaccard <- data.frame()
  for (i in 1:length(query_marker_cluster)) {
    for (j in 1:length(reference_marker_cluster)) {
      marker_jaccard[i, j] <- jaccard(query_marker_cluster[[i]]$gene, 
        reference_marker_cluster[[j]]$gene)
    }
  }
  rownames(marker_jaccard) <- names(query_marker_cluster)
  colnames(marker_jaccard) <- names(reference_marker_cluster)
  marker_jaccard[marker_jaccard == "NaN"] <- 0
  reference_all_means <- embedding[1:ncol(reference_matrix), 
    ]
  reference_all_means <- reference_all_means[, -1]
  reference_all_means <- as.data.frame(lapply(reference_all_means, 
    as.numeric))
  reference_all_means_vec <- colMeans(reference_all_means)
  query_all_means <- embedding[(ncol(reference_matrix) + 1):nrow(embedding), 
    ]
  query_all_means <- query_all_means[, -1]
  query_all_means <- as.data.frame(lapply(query_all_means, 
    as.numeric))
  query_all_means_vec <- colMeans(query_all_means)
  #Epsilon <- cor(reference_all_means_vec, query_all_means_vec)
  Epsilon <- cor(as.numeric(reference_all_means_vec),
               as.numeric(query_all_means_vec))
  cat("START4\n")
  beta <- max(CCA_PCC)/max(marker_jaccard)
  Epsilon <- max(Epsilon, random_PCC)
  CCA_marker_similarity <- CCA_PCC + beta * marker_jaccard
  CCA_marker_similarity[CCA_marker_similarity < Epsilon] <- 0
  match_matrix <- lol2(data = CCA_marker_similarity, top = 3, 
    threshold = Epsilon)
  match_matrix1 <- match_matrix
  match_matrix[match_matrix == 2] <- 1
  match_matrix <- t(as.matrix(match_matrix))
  reference_adj_matrix <- matrix(0, nrow = nrow(match_matrix), 
    ncol = nrow(reference@meta.data), dimnames = list(c(rownames(match_matrix)), 
      c(rownames(reference@meta.data))))
  for (i in 1:ncol(reference_adj_matrix)) {
    cluster_id <- reference@meta.data$seurat_clusters[i]
    j <- which(rownames(match_matrix) == cluster_id)
    reference_adj_matrix[j, i] <- 1
  }
  query_adj_matrix <- matrix(0, nrow = ncol(match_matrix), 
    ncol = nrow(query@meta.data), dimnames = list(c(colnames(match_matrix)), 
      c(rownames(query@meta.data))))
  for (i in 1:ncol(query_adj_matrix)) {
    cluster_id <- query@meta.data$seurat_clusters[i]
    j <- which(colnames(match_matrix) == cluster_id)
    query_adj_matrix[j, i] <- 1
  }
  cat("START5\n")
  reference_adj_matrix <- t(reference_adj_matrix)
  query_adj_matrix <- t(query_adj_matrix)
  reference_cell_embedding <- embedding[1:nrow(reference@meta.data), 
    ]
  reference_cell_embedding <- reference_cell_embedding[, -1]
  reference_cell_embedding <- apply(as.matrix(reference_cell_embedding), 
    2, as.numeric)
  rownames(reference_cell_embedding) <- rownames(embedding)[1:nrow(reference@meta.data)]
  query_cell_embedding <- embedding[(nrow(reference@meta.data) + 
    1):nrow(embedding), ]
  query_cell_embedding <- query_cell_embedding[, -1]
  query_cell_embedding <- apply(as.matrix(query_cell_embedding), 
    2, as.numeric)
  rownames(query_cell_embedding) <- rownames(embedding)[(nrow(reference@meta.data) + 
    1):nrow(embedding)]
  ones_matrix <- matrix(1, nrow = nrow(match_matrix), ncol = ncol(match_matrix))
  reference_means <- as.matrix(reference_means)
  query_means <- as.matrix(query_means)
  reference_means1 <- reference_means + query_means %*% t(distance_diff * 
    (match_matrix - ones_matrix))
  query_means1 <- query_means + reference_means %*% (distance_diff * 
    (match_matrix - ones_matrix))
  reference_means2 <- reference_means1 + query_means1 %*% 
    t(distance_same * match_matrix)
  query_means2 <- query_means1 + reference_means1 %*% (distance_same * 
    match_matrix)
  reference_cell_embedding1 <- reference_cell_embedding + 
    reference_adj_matrix %*% t(reference_means2 - reference_means)
  query_cell_embedding1 <- query_cell_embedding + query_adj_matrix %*% 
    t(query_means2 - query_means)
  cat("START6\n")
  L2norm <- function(data) {
    norm <- c()
    L2Data <- data
    z <- t(data) %*% data
    for (i in 1:ncol(data)) {
      norm[i] <- sqrt(z[i, i])
    }
    for (i in 1:ncol(data)) {
      L2Data[, i] <- data[, i]/norm[i]
    }
    rownames(L2Data) <- rownames(data)
    colnames(L2Data) <- colnames(data)
    return(L2Data)
  }
  cat("START7\n")
  cluster_means <- cbind(reference_means2, query_means2)
  L2_cluster_means <- L2norm(cluster_means)
  cell_embedding <- rbind(reference_cell_embedding1, query_cell_embedding1)
  L2_cell_embedding <- t(L2norm(t(cell_embedding)))
  cell_correct_embedding <- L2_cell_embedding %*% L2_cluster_means
  ClusterMatch_integration <- list(D1_reference = reference, 
    D2_query = query, Matching_matrix = match_matrix1, cell_embedding = cell_correct_embedding)
  return(ClusterMatch_integration)
}

lol2 <- function (data, top = 3, threshold = 1.96) 
{
  data <- as.matrix(data)
  match_matrix <- data
  for (i in 1:nrow(data)) {
    for (j in 1:ncol(data)) {
      top_row <- data[i, order(data[i, ], decreasing = TRUE)[1:top]]
      top_col <- data[order(data[, j], decreasing = TRUE)[1:top], 
        j]
      if (data[i, j] %in% top_row & data[i, j] %in% top_col & 
        data[i, j] > threshold) {
        match_matrix[i, j] <- 1
      }
      else match_matrix[i, j] <- 0
    }
  }
  for (i in 1:nrow(data)) {
    for (j in 1:ncol(data)) {
      top_row <- data[i, order(data[i, ], decreasing = TRUE)[1]]
      top_col <- data[order(data[, j], decreasing = TRUE)[1], 
        j]
      if (data[i, j] %in% top_row & data[i, j] %in% top_col & 
        data[i, j] > threshold) {
        match_matrix[i, j] <- 2
      }
    }
  }
  return(match_matrix)
}

```

```{r}
library(devtools)
library(ClusterMatch)
library(ggplot2)

setwd("/Users/wmw0016")
memory.limit(size = 56000) # set memory for windows 
dendritic_batch1 <- read.csv("./scrna_expression.csv", row.names = 1)
dendritic_batch2 <- read.csv("./spatial_expression.csv", row.names = 1)
dendritic_celltype <- read.csv("./celltype.csv")

dendritic_res <- ClusterMatch_resolution(dendritic_batch1, dendritic_batch2, ref.norm = TRUE, que.norm = TRUE)

#dendritic_matching <- ClusterMatch_matching(dendritic_batch1, dendritic_batch2, ref.res = dendritic_res$D1_ref_res, que.res = dendritic_res$D2_que_res, ref.norm = TRUE, que.norm = TRUE, random_PCC = 1.3)

dendritic_integration <- lol(dendritic_batch1, dendritic_batch2, ref.res = dendritic_res$D1_ref_res,
                                                  que.res = dendritic_res$D2_que_res, ref.norm = TRUE, que.norm = TRUE, random_PCC = 1.3, distance_diff = 3, distance_same = 1)

umap_df <- ClusterMatch_UMAP(embedding = dendritic_integration$cell_embedding, celltype = dendritic_celltype)

batch_colour=c("#E64540", "#3F81BB")
celltype_colour=c("#E64136", "#5F78A3", "#EDA6C3", "#96C561")

ggplot(umap_df,aes(X1, X2, color = batch)) + 
  scale_color_manual(values = batch_colour) +
  geom_point() + theme_bw() +
  theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5), text = element_text(size = 20)) +
  labs(x="UMAP_1", y="UMAP_2", title = "ClusterMatch")

ggplot(umap_df,aes(X1, X2, color = label)) + 
  scale_color_manual(values = celltype_colour) +
  geom_point() + theme_bw() +
  theme(panel.grid=element_blank(), plot.title = element_text(hjust = 0.5), text = element_text(size = 20)) +
  labs(x="UMAP_1", y="UMAP_2", title = "ClusterMatch")
```




