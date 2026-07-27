

```{r}
library(ggplot2)
library(patchwork)

set.seed(1)

## ---- 1. Cells in 2D, 4 clusters ---------------------------
n_per   <- 200
centers <- data.frame(cluster = factor(1:4),
                       cx = c(0, 5, 0, 5),
                       cy = c( 5, 5, 0, 0))

cells <- do.call(rbind, lapply(1:4, function(k) {
  data.frame(cluster = factor(k),
             x = rnorm(n_per, centers$cx[k], 1),
             y = rnorm(n_per, centers$cy[k], 1))
}))
cells$id <- seq_len(nrow(cells))

## ---- palette + shared theme -------------------------------
pal <- c("1" = "#E69F00", "2" = "#56B4E9", "3" = "#009E73", "4" = "#CC79A7")
grey_pal <- c("1" = "grey", "2" = "grey", "3" = "grey", "4" = "grey")
base_theme <- theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        legend.position  = "none",
        plot.title       = element_text(face = "bold", size = 11))

## ---- 2. "GLM-PCA" embedding (just a rotated/jittered view) -
theta <- pi / 6
R <- matrix(c(cos(theta), -sin(theta), sin(theta), cos(theta)), 2)
emb <- as.matrix(cells[, c("x", "y")]) %*% R
cells$pc1 <- emb[, 1] * 0.9 + rnorm(nrow(cells), 0, 0.3)
cells$pc2 <- emb[, 2] * 1.1 + rnorm(nrow(cells), 0, 0.3)

## ---- 3. Selection (illustrative only) ---------------------
coords <- as.matrix(cells[, c("pc1", "pc2")])

# D: representative per cluster = cell nearest its cluster centroid (labels 1-4)
reps <- do.call(rbind, lapply(1:4, function(k) {
  s   <- cells[cells$cluster == k, ]
  cen <- colMeans(s[, c("pc1", "pc2")])
  s[which.min((s$pc1 - cen[1])^2 + (s$pc2 - cen[2])^2), ]
}))
reps$order_lab <- 1:4

# E: iterative pick = closest cell to the centroid cell that is still at
#    least `min_gap` away, so the marker/label stays visible (labels 5-8)
min_gap <- 0.9
extra <- do.call(rbind, lapply(1:4, function(k) {
  rid  <- reps$id[k]
  cand <- cells[cells$cluster == k & cells$id != rid, ]
  d    <- sqrt((cand$pc1 - coords[rid, 1])^2 + (cand$pc2 - coords[rid, 2])^2)
  cand <- cand[d >= min_gap, ]; d <- d[d >= min_gap]
  cand[which.min(d), ]                 # nearest cell beyond the gap
}))
extra$order_lab <- 5:8

sel_all <- rbind(reps, extra)   # all labeled, selected cells

## ---- 4. Panels --------------------------------------------
pA <- ggplot(cells, aes(x, y, color = cluster)) +
  geom_point(size = 1.6, alpha = 0.85) +
  scale_color_manual(values = grey_pal) +
  labs(title = "", x = "Dim 1", y = "Dim 2") +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),       # Removes all major and minor grid lines
    axis.ticks = element_blank(),       # Removes all axis tick marks
    axis.text = element_blank(),        # Removes the text labels next to the ticks
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.title = element_blank()        # (Optional) Removes axis titles (e.g., "wt", "mpg")
  )

pB <- ggplot(cells, aes(pc1, pc2, color = cluster)) +
  geom_point(size = 1.6, alpha = 0.85) +
  scale_color_manual(values = grey_pal) +
  labs(title = "", x = "Dim 1", y = "Dim 2") +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),       # Removes all major and minor grid lines
    axis.ticks = element_blank(),       # Removes all axis tick marks
    axis.text = element_blank(),        # Removes the text labels next to the ticks
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.title = element_blank()        # (Optional) Removes axis titles (e.g., "wt", "mpg")
  )

pC <- ggplot(cells, aes(pc1, pc2, color = cluster)) +
  stat_ellipse(type = "norm", level = 0.9, linewidth = 0.6) +
  geom_point(size = 1.6, alpha = 0.85) +
  scale_color_manual(values = pal, name = NULL,
                     labels = paste("Cluster", 1:4)) +
  labs(title = "", x = "Dim 1", y = "Dim 2") +
  theme_minimal() + theme(legend.position = "right") +
  theme(
    panel.grid = element_blank(),       # Removes all major and minor grid lines
    axis.ticks = element_blank(),       # Removes all axis tick marks
    axis.text = element_blank(),        # Removes the text labels next to the ticks
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.title = element_blank()        # (Optional) Removes axis titles (e.g., "wt", "mpg")
  )

pD <- ggplot(cells, aes(pc1, pc2)) +
  geom_point(color = "grey80", size = 1.4) +
  geom_point(data = reps, aes(fill = cluster),
             shape = 21, color = "black", size = 3.8, stroke = 0.8) +
  geom_text(data = reps, aes(label = order_lab), size = 2.6, vjust = -1.1) +
  scale_fill_manual(values = pal) +
  labs(title = "",
       x = "Dim 1", y = "Dim 2") +
  theme_minimal() + 
  theme(
    panel.grid = element_blank(),       # Removes all major and minor grid lines
    axis.ticks = element_blank(),       # Removes all axis tick marks
    axis.text = element_blank(),        # Removes the text labels next to the ticks
    axis.title = element_blank(),        # (Optional) Removes axis titles (e.g., "wt", "mpg")
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
  )

pE <- ggplot(cells, aes(pc1, pc2)) +
  geom_point(color = "grey80", size = 1.4) +
  geom_point(data = reps,  aes(fill = cluster),          # centroid cells (squares)
             shape = 22, color = "black", size = 3.6, stroke = 0.8) +
  geom_point(data = extra, aes(fill = cluster),          # nearby picks (circles)
             shape = 21, color = "black", size = 3.4, stroke = 0.8) +
  geom_text(data = sel_all, aes(label = order_lab), size = 2.6, vjust = -1.1) +
  scale_fill_manual(values = pal) +
  labs(title = "", x = "GLM-PC1", y = "GLM-PC2") +
  theme_minimal() + 
  theme(
    panel.grid = element_blank(),       # Removes all major and minor grid lines
    axis.ticks = element_blank(),       # Removes all axis tick marks
    axis.text = element_blank(),        # Removes the text labels next to the ticks
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.title = element_blank()        # (Optional) Removes axis titles (e.g., "wt", "mpg")
  )

## ---- 5. Assemble + save -----------------------------------
fig <- (pB | pC) / (pD | pE)

ggsave("method_overview.png", fig, width = 12, height = 12, dpi = 300)
ggsave("method_overview.pdf", fig, width = 12, height = 12)
fig

## ---- Per-cluster highlight: grey varies within & across panels ----
low_grey  <- c("grey10", "grey20", "grey30", "grey40")   # darkest point per panel
high_grey <- c("grey45", "grey55", "grey65", "grey75")   # lightest point per panel

highlight_plots <- lapply(1:4, function(k) {
  bg <- subset(cells, cluster != k)
  fg <- subset(cells, cluster == k)
  cen <- colMeans(fg[, c("pc1", "pc2")])
  fg$d <- sqrt((fg$pc1 - cen[1])^2 + (fg$pc2 - cen[2])^2)   # distance to centroid

  ggplot() +
    geom_point(data = bg, aes(pc1, pc2), color = "grey85", size = 1.4) +
    geom_point(data = fg, aes(pc1, pc2, color = d), size = 1.7) +
    scale_color_gradient(low = low_grey[k], high = high_grey[k], guide = "none") +
    labs(title = paste("GLM_PC", k), x = "GLM-PC1", y = "GLM-PC2") +
    theme_minimal() + 
    theme(
    panel.grid = element_blank(),       # Removes all major and minor grid lines
    axis.ticks = element_blank(),       # Removes all axis tick marks
    axis.text = element_blank(),        # Removes the text labels next to the ticks
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.title = element_blank()        # (Optional) Removes axis titles (e.g., "wt", "mpg")
  )
})

fig_clusters <- patchwork::wrap_plots(highlight_plots, ncol = 2)

ggsave("clusters_highlight.png", fig_clusters, width = 9, height = 8, dpi = 300)
ggsave("clusters_highlight.pdf", fig_clusters, width = 9, height = 8)
fig_clusters
```

```{r}
set.seed(42)
frac <- 0.30
cells$assigned <- runif(nrow(cells)) < frac
cells$ctype <- factor(ifelse(cells$assigned, as.character(cells$cluster), "Unassigned"),
                      levels = c("1", "2", "3", "4", "Unassigned"))
pal2 <- c(pal, "Unassigned" = "grey80")        # cluster colors + grey for unassigned
# draw unassigned first (underneath), assigned cells on top
cells_ord <- cells[order(cells$ctype == "Unassigned", decreasing = TRUE), ]
p_partial <- ggplot(cells_ord, aes(pc1, pc2, color = ctype)) +
  geom_point(size = 1.6, alpha = 0.9) +
  scale_color_manual(values = pal2, name = NULL,
                     breaks = c("1", "2", "3", "4", "Unassigned"),
                     labels = c(paste("Cluster", 1:4), "Unassigned")) +
  labs(title = "Partial clustering (~70% of cells assigned)",
       x = "GLM-PC1", y = "GLM-PC2") +
  theme(legend.position = "right") + 
  theme_minimal() + 
    theme(
    panel.grid = element_blank(),       # Removes all major and minor grid lines
    axis.ticks = element_blank(),       # Removes all axis tick marks
    axis.text = element_blank(),        # Removes the text labels next to the ticks
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.title = element_blank()        # (Optional) Removes axis titles (e.g., "wt", "mpg")
  )
ggsave("partial_clustering.png", p_partial, width = 6.5, height = 5, dpi = 300)
ggsave("partial_clustering.pdf", p_partial, width = 6.5, height = 5)
p_partial
```

```{r}
## ---- Connectivity: tunable within / between edge fractions ----
set.seed(7)
line_col <- "grey40"        # one color for ALL lines

within_frac  <- 0.6         # bold within-cluster edges  (× #assigned cells)
between_frac <- 0.2         # thin between-cluster edges  (× #assigned cells)
thin_frac    <- 0.4         # thin unassigned->assigned   (× #unassigned cells)

asg <- cells[cells$assigned, ]     # assigned (highlighted) cells
un  <- cells[!cells$assigned, ]    # unassigned cells

## sample n assigned-assigned pairs of a given type (same cluster or not)
sample_pairs <- function(n, same) {
  if (n <= 0) return(data.frame(x = numeric(), y = numeric(),
                                xend = numeric(), yend = numeric()))
  out <- data.frame()
  while (nrow(out) < n) {
    m  <- (n - nrow(out)) * 4 + 10
    a  <- sample(nrow(asg), m, replace = TRUE)
    b  <- sample(nrow(asg), m, replace = TRUE)
    ok <- (a != b) & ((asg$cluster[a] == asg$cluster[b]) == same)
    a  <- a[ok]; b <- b[ok]
    out <- rbind(out, data.frame(x = asg$pc1[a],  y = asg$pc2[a],
                                 xend = asg$pc1[b], yend = asg$pc2[b]))
  }
  out[seq_len(n), ]
}

bold_edges <- sample_pairs(round(nrow(asg) * within_frac),  same = TRUE)   # within  -> bold
betw_edges <- sample_pairs(round(nrow(asg) * between_frac), same = FALSE)  # between -> thin

## thin: random subset of UNASSIGNED cells linked to nearest assigned cell
pick <- sample(nrow(un), round(nrow(un) * thin_frac))
thin_un <- do.call(rbind, lapply(pick, function(i) {
  d <- (asg$pc1 - un$pc1[i])^2 + (asg$pc2 - un$pc2[i])^2
  j <- which.min(d)
  data.frame(x = un$pc1[i], y = un$pc2[i],
             xend = asg$pc1[j], yend = asg$pc2[j])
}))

thin_edges <- rbind(betw_edges, thin_un)   # between-cluster + unassigned, all thin

## ---- plot ------------------------------------------------
p_net <- ggplot() +
  geom_segment(data = thin_edges, aes(x, y, xend = xend, yend = yend),
               color = line_col, linewidth = 0.3, alpha = 0.6) +          # thin
  geom_segment(data = bold_edges, aes(x, y, xend = xend, yend = yend),
               color = line_col, linewidth = 1.1, alpha = 0.9,
               lineend = "round") +                                       # bold
  geom_point(data = cells_ord, aes(pc1, pc2, color = ctype),
             size = 1.6, alpha = 0.9) +                                    # same highlight
  scale_color_manual(values = pal2, name = NULL,
                     breaks = c("1", "2", "3", "4", "Unassigned"),
                     labels = c(paste("Cluster", 1:4), "Unassigned")) +
  labs(title = "Cell connectivity", x = "GLM-PC1", y = "GLM-PC2") +
  theme_minimal() +
  theme(panel.grid    = element_blank(),
        axis.ticks    = element_blank(),
        axis.text     = element_blank(),
        axis.title    = element_blank(),
        panel.border  = element_rect(color = "black", fill = NA, linewidth = 1),
        legend.position = "right")

ggsave("cell_connectivity.png", p_net, width = 6.5, height = 5, dpi = 300)
ggsave("cell_connectivity.pdf", p_net, width = 6.5, height = 5)
p_net

```

```{r}

## ---- Unassigned cells colored by proximity-weighted cluster ----
set.seed(11)
p_near <- 0.5        # chance an unassigned cell takes its NEAREST cluster's color

asg <- cells[cells$assigned, ]
# cluster centroids from the assigned (known) cells
cen <- t(sapply(1:4, function(k) colMeans(asg[asg$cluster == k, c("pc1", "pc2")])))

# display cluster: true cluster if assigned, proximity-sampled if unassigned
cells$dcluster <- as.character(cells$cluster)
for (i in which(!cells$assigned)) {
  d    <- sqrt((cen[, 1] - cells$pc1[i])^2 + (cen[, 2] - cells$pc2[i])^2)
  near <- which.min(d)
  prob <- numeric(4)
  prob[near] <- p_near                              # ~50% to nearest cluster
  oth  <- setdiff(1:4, near)
  prob[oth] <- (1 - p_near) * (1 / d[oth]) / sum(1 / d[oth])   # rest by closeness
  cells$dcluster[i] <- as.character(sample(1:4, 1, prob = prob))
}
cells$dcluster <- factor(cells$dcluster, levels = c("1", "2", "3", "4"))

## ---- plot ------------------------------------------------
p_fill <- ggplot() +
  geom_point(data = subset(cells, !assigned),                 # unassigned: faded
             aes(pc1, pc2, color = dcluster), size = 1.5, alpha = 0.7) +
  geom_point(data = subset(cells, assigned),                  # assigned: highlighted
             aes(pc1, pc2, color = dcluster), size = 1.7, alpha = 1.0) +
  scale_color_manual(values = pal, name = NULL, labels = paste("Cluster", 1:4)) +
  labs(title = "Unassigned cells colored by proximity-weighted cluster",
       x = "GLM-PC1", y = "GLM-PC2") +
  theme_minimal() +
  theme(panel.grid    = element_blank(),
        axis.ticks    = element_blank(),
        axis.text     = element_blank(),
        axis.title    = element_blank(),
        panel.border  = element_rect(color = "black", fill = NA, linewidth = 1),
        legend.position = "right")

ggsave("proximity_fill.png", p_fill, width = 6.5, height = 5, dpi = 300)
ggsave("proximity_fill.pdf", p_fill, width = 6.5, height = 5)
p_fill
```

