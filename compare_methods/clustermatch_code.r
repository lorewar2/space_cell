library(devtools)
library(ClusterMatch)
library(ggplot2)

setwd("/Users/wmw0016")
dendritic_batch1 <- read.csv("./scrna_expression.csv", row.names = 1)
dendritic_batch2 <- read.csv("./spatial_expression.csv", row.names = 1)
dendritic_celltype <- read.csv("./celltype.csv")

dendritic_res <- ClusterMatch_resolution(dendritic_batch1, dendritic_batch2, ref.norm = FALSE, que.norm = FALSE)

dendritic_matching <- ClusterMatch_matching(dendritic_batch1, dendritic_batch2, ref.res = dendritic_res$D1_ref_res, que.res = dendritic_res$D2_que_res, ref.norm = FALSE, que.norm = FALSE, random_PCC = 1.3)

dendritic_integration <- ClusterMatch_integration(dendritic_batch1, dendritic_batch2, ref.res = dendritic_res$D1_ref_res,
                                                  que.res = dendritic_res$D2_que_res, ref.norm = FALSE, que.norm = FALSE, random_PCC = 1.3, distance_diff = 3, distance_same = 1)

umap_df <- ClusterMatch_UMAP(embedding = dendritic_integration$cell_embedding, celltype = dendritic_celltype)

batch_colour=c("#E64540","#3F81BB")
celltype_colour=c("#E64136","#5F78A3","#EDA6C3","#96C561")

ggplot(umap_df,aes(X1,X2,color=batch)) + 
  scale_color_manual(values = batch_colour)+
  geom_point() + theme_bw() +
  theme(panel.grid=element_blank(),plot.title = element_text(hjust = 0.5),text = element_text(size = 20)) +
  labs(x="UMAP_1",y="UMAP_2",
       title = "ClusterMatch")

ggplot(umap_df,aes(X1,X2,color=label)) + 
  scale_color_manual(values = celltype_colour)+
  geom_point() + theme_bw() +
  theme(panel.grid=element_blank(),plot.title = element_text(hjust = 0.5),text = element_text(size = 20)) +
  labs(x="UMAP_1",y="UMAP_2",
       title = "ClusterMatch")