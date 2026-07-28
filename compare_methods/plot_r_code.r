

```{r}
library(ggplot2)
library(patchwork)
library(MASS)
library(ggforce)

set.seed(1)

## Number of cells per cluster
n_per <- 200

## Define cluster parameters
clusters <- list(

  list(
    name   = "4",
    center = c(1, 7),
    cell_number = 200,
    Sigma  = matrix(c(0.5, 0,
                      0, 0.5), nrow = 2)   # positive diagonal
  ),
  list(
    name   = "1-1",
    center = c(4.4, 6),
    cell_number = 200,
    Sigma  = matrix(c(0.8, 0.6,
                      0.6, 0.8), nrow = 2)   # positive diagonal
  ),
  list(
    name   = "1-2",
    center = c(1, 3),
    cell_number = 200,
    Sigma  = matrix(c(0.3, 0.1,
                      0.1, 0.3), nrow = 2)   # positive diagonal
  ),
  list(
    name   = "2-1",
    center = c(9, 8),
    cell_number = 200,
    Sigma  = matrix(c(0.4, 0.0,
                      0.0, 0.4), nrow = 2)   # positive diagonal
  ),
  list(
    name   = "2-2",
    center = c(7.5, 4.8),
    cell_number = 200,
    Sigma  = matrix(c(0.3, 0.1,
                      0.2, 0.3), nrow = 2)   # positive diagonal
  ),
  list(
    name   = "2-3",
    center = c(0.5, 0.5),
    cell_number = 50,
    Sigma  = matrix(c(1, 0.1,
                      0.2, 0.1), nrow = 2)   # positive diagonal
  ),
  list(
    name   = "3",
    center = c(5.5, 0.5),
    cell_number = 200,
    Sigma  = matrix(c(0.9, -0.6,
                      -0.6, 0.9), nrow = 2)   # positive diagonal
  ),
  list(
    name   = "1-3",
    center = c(9.5, 3.8),
    cell_number = 50,
    Sigma  = matrix(c(0.1, 0,
                      0, 1.0), nrow = 2)   # positive diagonal
  )

)

## Generate cells
cells <- do.call(rbind, lapply(seq_along(clusters), function(k) {

  xy <- mvrnorm(
    n = clusters[[k]]$cell_number,
    mu = clusters[[k]]$center,
    Sigma = clusters[[k]]$Sigma
  )

  data.frame(
    id           = paste0("cell_", (k - 1) * clusters[[k]]$cell_number + seq_len(n_per)),
    true_cluster = clusters[[k]]$name,
    x            = xy[, 1],
    y            = xy[, 2]
  )
}))

cells$id <- seq_len(nrow(cells))

## ---- palette + shared theme -------------------------------
pal <- c("1" = "#D85A30", "2" = "#1D9E75", "3" = "#7F77DD", "4" = "#EF9F27")
grey_pal <- c("1-1" = "#D85A30", "1-2" = "#D85A30", "1-3" = "#D85A30","2-1" = "#1D9E75", "2-2" = "#1D9E75","2-3" = "#1D9E75","3" = "#7F77DD", "4" = "#EF9F27")
base_theme <- theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        legend.position  = "none",
        plot.title       = element_text(face = "bold", size = 11))

ggplot(cells, aes(x, y, color = true_cluster)) +
  geom_point(size = 1.6, alpha = 0.85) +
  scale_color_manual(values = grey_pal) +
  labs(title = "", x = "Dim 1", y = "Dim 2") +
  coord_cartesian(xlim = c(0, 10), ylim = c(0, 10)) +
  theme_classic() +
  theme(
    panel.grid = element_blank(),       # Removes all major and minor grid lines
    axis.ticks = element_blank(),       # Removes all axis tick marks
    axis.text = element_blank(),        # Removes the text labels next to the ticks
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.title = element_blank(),       # (Optional) Removes axis titles (e.g., "wt", "mpg"),
    legend.position = "none"
  )
```
```{r}
## ---- helper: highlight chosen clusters dark grey, rest light grey ----
plot_highlight <- function(cells, highlight, title = "") {
  cells$hl <- ifelse(cells$true_cluster %in% highlight, "yes", "no")

  ggplot(cells, aes(x, y)) +
    # background: non-highlighted cells in light grey
    geom_point(data = subset(cells, hl == "no"),
               color = "grey80", size = 1.6, alpha = 0.6) +
    # foreground: highlighted cells in dark grey
    geom_point(data = subset(cells, hl == "yes"),
               color = "grey25", size = 1.6, alpha = 0.9) +
    labs(title = title, x = "Dim 1", y = "Dim 2") +
    coord_cartesian(xlim = c(0, 10), ylim = c(0, 10)) +
    theme_classic() +
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      panel.grid = element_blank(),
      axis.ticks = element_blank(),
      axis.text  = element_blank(),
      axis.title = element_blank(),
      plot.title = element_text(face = "bold", size = 11),
      legend.position = "none"
    )
}

## ---- define which clusters each plot highlights ----------------------
p1 <- plot_highlight(cells, c("1-1", "1-2", "1-3"), title = "GLM-PC1")
p2 <- plot_highlight(cells, c("2-1", "2-2", "2-3"), title = "GLM-PC2")
p3 <- plot_highlight(cells, c("3"),                 title = "GLM-PC3")
p4 <- plot_highlight(cells, c("4"),                 title = "GLM-PC4")

## ---- assemble 2x2 grid -----------------------------------------------
(p1 | p2) / (p3 | p4)
```

```{r}
library(ggforce)
## ---- helper: color highlighted clusters, circle each sub-cluster -----
plot_highlight_circled <- function(cells, highlight, group_key, title = "") {
  # group_key = "1", "2", "3", or "4" → picks the palette colour
  col <- pal[[group_key]]

  hl_cells  <- subset(cells, true_cluster %in% highlight)
  bg_cells  <- subset(cells, !(true_cluster %in% highlight))

  ggplot() +
    # background: non-highlighted cells in light grey
    geom_point(data = bg_cells, aes(x, y),
               color = "grey85", size = 1.6, alpha = 0.6) +
    # foreground: highlighted cells in the group colour
    geom_point(data = hl_cells, aes(x, y),
               color = col, size = 1.6, alpha = 0.9) +
    # circle each sub-cluster individually (grouped by true_cluster)
    geom_mark_ellipse(data = hl_cells,
                      aes(x, y, group = true_cluster),
                      color = col, fill = NA,
                      linewidth = 0.7, expand = unit(2, "mm")) +
    labs(title = title, x = "Dim 1", y = "Dim 2") +
    coord_cartesian(xlim = c(0, 10), ylim = c(0, 10)) +
    theme_classic() +
    theme(
      panel.grid = element_blank(),
      axis.ticks = element_blank(),
      axis.text  = element_blank(),
      axis.title = element_blank(),
      plot.title = element_text(face = "bold", size = 11),
      legend.position = "none"
    )
}

## ---- one plot per group ----------------------------------------------
p1 <- plot_highlight_circled(cells, c("1-1", "1-2", "1-3"), "1", "Cluster 1")
p2 <- plot_highlight_circled(cells, c("2-1", "2-2", "2-3"), "2", "Cluster 2")
p3 <- plot_highlight_circled(cells, c("3"),                 "3", "Cluster 3")
p4 <- plot_highlight_circled(cells, c("4"),                 "4", "Cluster 4")

## ---- stack horizontally ----------------------------------------------
p1 / p2 / p3 / p4

```

```{r}

set.seed(102)
## ---- map each sub-cluster to its parent group ------------------------
sub_to_group <- c(
  "1-1" = "1", "1-2" = "1", "1-3" = "5",
  "2-1" = "2", "2-2" = "2", "2-3" = "5",
  "3"   = "3",
  "4"   = "4"
)

## tag every cell with its parent group
cells$group <- sub_to_group[as.character(cells$true_cluster)]

## ---- pick ONE random cell per parent group (4 cells total) ----------
selected <- do.call(rbind, lapply(unique(cells$group), function(g) {
  pool <- cells[cells$group == g, ]
  pool[sample(nrow(pool), 1), ]
}))

## ---- plot: all cells grey, selected cells in group colour -----------
ggplot() +
  geom_point(data = cells, aes(x, y),
             color = "grey85", size = 1.6, alpha = 0.6) +
  geom_point(data = selected, aes(x, y, fill = group),
             shape = 21, color = "black", size = 2.5, stroke = 1, alpha = 1) +
  scale_fill_manual(values = pal) +
  labs(title = "", x = "Dim 1", y = "Dim 2") +
  coord_cartesian(xlim = c(0, 10), ylim = c(0, 10)) +
  theme_classic() +
  theme(
    panel.grid = element_blank(),
    axis.ticks = element_blank(),
    axis.text  = element_blank(),
    axis.title = element_blank(),
    legend.position = "none"
  )
```
```{r}
set.seed(102)
## ---- first selection: one cell per group (circles) -------------------
selected_circle <- do.call(rbind, lapply(unique(cells$group), function(g) {
  pool <- cells[cells$group == g, ]
  pool[sample(nrow(pool), 1), ]
}))

## ---- second selection: a fairly close neighbour (squares) ------------
## for each circle cell, rank same-group cells by distance and pick one
## from a "fairly close" band (not the nearest, not far)
selected_square <- do.call(rbind, lapply(seq_len(nrow(selected_circle)), function(i) {
  ref  <- selected_circle[i, ]
  pool <- cells[cells$group == ref$group & cells$id != ref$id, ]

  d <- sqrt((pool$x - ref$x)^2 + (pool$y - ref$y)^2)
  ord <- order(d)

  # take from ranks ~5-10 (fairly close, but not immediately adjacent)
  band <- ord[min(5, length(ord)):min(10, length(ord))]
  pool[sample(band, 1), ]
}))

## ---- plot ------------------------------------------------------------
ggplot() +
  geom_point(data = cells, aes(x, y),
             color = "grey85", size = 1.6, alpha = 0.6) +
  geom_point(data = selected_circle, aes(x, y, fill = group),
             shape = 21, color = "black", size = 2.5, stroke = 1) +
  geom_point(data = selected_square, aes(x, y, fill = group),
             shape = 22, color = "black", size = 2.5, stroke = 1) +
  scale_fill_manual(values = pal) +
  labs(title = "", x = "Dim 1", y = "Dim 2") +
  coord_cartesian(xlim = c(0, 10), ylim = c(0, 10)) +
  theme_classic() +
  theme(
    panel.grid = element_blank(),
    axis.ticks = element_blank(),
    axis.text  = element_blank(),
    axis.title = element_blank(),
    legend.position = "none"
  )
```

```{r}
set.seed(2)
## ---- map each sub-cluster to its parent group ------------------------
sub_to_group <- c(
  "1-1" = "1", "1-2" = "1", "1-3" = "1",
  "2-1" = "2", "2-2" = "2", "2-3" = "2",
  "3"   = "3",
  "4"   = "4"
)
cells$group <- sub_to_group[as.character(cells$true_cluster)]

## sub-clusters to leave grey regardless
exclude_sub <- c("1-3", "2-3")

## ---- pick 30% of cells from each parent group (excluding 1-3, 2-3) ---
chosen <- do.call(rbind, lapply(unique(cells$group), function(g) {
  pool <- cells[cells$group == g & !(cells$true_cluster %in% exclude_sub), ]
  n_pick <- ceiling(0.30 * nrow(pool))
  pool[sample(nrow(pool), n_pick), ]
}))

## ---- plot: chosen cells coloured, everything else grey --------------
ggplot() +
  # background: all cells grey (includes unchosen + excluded 1-3, 2-3)
  geom_point(data = cells, aes(x, y),
             color = "grey85", size = 1.6, alpha = 0.6) +
  # foreground: chosen 30% coloured by group
  geom_point(data = chosen, aes(x, y, color = group),
             size = 1.8, alpha = 0.95) +
  scale_color_manual(values = pal) +
  labs(title = "", x = "Dim 1", y = "Dim 2") +
  coord_cartesian(xlim = c(0, 10), ylim = c(0, 10)) +
  theme_classic() +
  theme(
    panel.grid = element_blank(),
    axis.ticks = element_blank(),
    axis.text  = element_blank(),
    axis.title = element_blank(),
    legend.position = "none"
  )
```

```{r}
set.seed(2)
cells$cluster <- sub_to_group[as.character(cells$true_cluster)]

## ---- define assigned cells: 30% per group, excluding 1-3 and 2-3 -----
exclude_sub <- c("1-3", "2-3")

cells$assigned <- FALSE
for (g in unique(cells$cluster)) {
  pool_idx <- which(cells$cluster == g & !(cells$true_cluster %in% exclude_sub))
  n_pick   <- ceiling(0.30 * length(pool_idx))
  chosen   <- sample(pool_idx, n_pick)
  cells$assigned[chosen] <- TRUE
}

## ctype: assigned cells keep their group colour, others = "Unassigned"
cells$ctype <- ifelse(cells$assigned, cells$cluster, "Unassigned")

pal2 <- c(pal, "Unassigned" = "grey80")

## draw unassigned first, assigned on top
cells_ord <- cells[order(cells$assigned), ]

## ---- Connectivity: tunable within / between edge fractions ----
line_col     <- "grey40"
within_frac  <- 0.3
between_frac <- 0.1
thin_frac    <- 0.4

asg <- cells[cells$assigned, ]
un  <- cells[!cells$assigned, ]

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
    out <- rbind(out, data.frame(x = asg$x[a],  y = asg$y[a],
                                 xend = asg$x[b], yend = asg$y[b]))
  }
  out[seq_len(n), ]
}

bold_edges <- sample_pairs(round(nrow(asg) * within_frac),  same = TRUE)
betw_edges <- sample_pairs(round(nrow(asg) * between_frac), same = FALSE)

## thin: random subset of UNASSIGNED cells linked to nearest assigned cell
pick <- sample(nrow(un), round(nrow(un) * thin_frac))
thin_un <- do.call(rbind, lapply(pick, function(i) {
  d <- (asg$x - un$x[i])^2 + (asg$y - un$y[i])^2
  j <- which.min(d)
  data.frame(x = un$x[i], y = un$y[i],
             xend = asg$x[j], yend = asg$y[j])
}))

thin_edges <- rbind(betw_edges, thin_un)

## ---- plot ------------------------------------------------------------
ggplot() +
  geom_segment(data = thin_edges, aes(x, y, xend = xend, yend = yend),
               color = line_col, linewidth = 0.3, alpha = 0.6) +
  geom_segment(data = bold_edges, aes(x, y, xend = xend, yend = yend),
               color = line_col, linewidth = 0.6, alpha = 0.9,
               lineend = "round") +
  geom_point(data = cells_ord, aes(x, y, color = ctype),
             size = 1.6, alpha = 0.9) +
  scale_color_manual(values = pal2, name = NULL,
                     breaks = c("1", "2", "3", "4", "Unassigned"),
                     labels = c(paste("Cluster", 1:4), "Unassigned")) +
  coord_cartesian(xlim = c(0, 10), ylim = c(0, 10)) +
  theme_classic() +
  theme(
    panel.grid = element_blank(),
    axis.ticks = element_blank(),
    axis.text  = element_blank(),
    axis.title = element_blank(),
    legend.position = "none"
  )


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

