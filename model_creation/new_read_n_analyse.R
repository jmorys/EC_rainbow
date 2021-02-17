library(progress)
library(tidyverse)
library(igraph)
library(ggplot2)
library(ggraph)
library(magrittr)
library(reshape2)
library(tidymodels)
library(foreach)
library(doParallel)
library(patchwork)
library(purrr)


graph_f_csvs <- function(dir){
  edgelist = read_csv(paste0(dir, "/edge_parametres.csv"), col_names = F)
  celllist = read_csv(paste0(dir, "/cell_parametres.csv"), col_names = F)
  
  colnames(edgelist) <- c("ori", "dest", "min", "mean", "max", "std")
  edgelist$ori %<>% as.integer
  edgelist$dest %<>% as.integer
  colnames(celllist) <- c("ID", "col", "y", "x", "area", "eccentricity")
  celllist$ID %<>% as.integer
  celllist$col %<>% as.factor
  
  
  edges_f_graph = as.matrix(edgelist[, 1:2])
  missing_nodes = max(edgelist$ori) < max(celllist$ID)
  
  if (missing_nodes) {
    edges_f_graph <- rbind(edges_f_graph, c(max(celllist$ID), max(celllist$ID)))
  }
  g = graph_from_edgelist(edges_f_graph)
  
  if (missing_nodes) {
    g = delete.edges(g, length(E(g))) 
  }
  
  
  celllist  %>% ggplot(aes(area, eccentricity)) + geom_bin2d()
  
  
  g = set.edge.attribute(g, "min", value = edgelist$min)
  g = set.edge.attribute(g, "mean", value =  edgelist$mean)
  g = set.edge.attribute(g, "max", value =  edgelist$max)
  g = set.edge.attribute(g, "std", value = edgelist$std)
  
  g <- set.vertex.attribute(g, "name", value = celllist$ID)
  g <- set.vertex.attribute(g, "colour", value = celllist$col)
  g <- set.vertex.attribute(g, "y", value = celllist$y)
  g <- set.vertex.attribute(g, "x", value = celllist$x)
  g <- set.vertex.attribute(g, "area", value = celllist$area)
  g <- set.vertex.attribute(g, "eccentricity", value = celllist$eccentricity)
  
  sigma = 140
  weight = exp(-(1 / (2 * sigma^2))*abs(edgelist$min)^2)
  
  g = set.edge.attribute(g, "weight", value = weight)
  return(g)
}


r = "D:/EC_rainbow_data/data_for_graphs"

dirs = list.dirs(r, recursive = F)

names <- modify(dirs,  function(x){
  path <- stringr::str_split(x, "/", simplify = TRUE)
  return(rev(path)[1])
  })
names(dirs) <- names
dirs <- as.list(dirs)

dirs <- modify(dirs,  function(x){
  graph_f_csvs(x)
})


targ = "D:/EC_rainbow_data/graphs"
if (!dir.exists(targ)) {
  dir.create(targ)
}

for (i in 1:length(dirs)) {
  saveRDS(dirs[[i]], file = paste0(targ, "/", names(dirs)[i], ".rds"))
}

g = dirs[[1]]


plot = ggraph(g, layout = tibble(get.vertex.attribute(g)$x, get.vertex.attribute(g)$y)) +
  geom_edge_link() + geom_node_point(size = 2, aes(color = as_factor(colour)), show.legend = F) + scale_y_reverse()
plot + ggsave("D:/EC_rainbow_data/test_plot.png",  type = "cairo", width = 9.5, height =  21.6)



colors = map(dirs, function(x){
  colour <- get.vertex.attribute(x, "colour")
  return(summary(as.factor(colour)))
})


dd  <-  as.data.frame(matrix(unlist(colors), nrow = length(unlist(colors[1]))))
dd <- apply(dd, MARGIN = 2, FUN = function(x){
  x/sum(x)
}) 
colnames(dd) <- stringr::str_extract(names(dirs), "[^_]+_[^_]+_[^_]+")
dd <- as.data.frame(dd) %>% add_column(color = c(1, 2, 3))
dd <- pivot_longer(dd, cols = colnames(dd)[-dim(dd)[2]])
dd %>% ggplot(aes(x = name, y = value, fill = as_factor(color))) + geom_col() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))



