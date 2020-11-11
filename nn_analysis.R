library(tidyverse)
library(igraph)
library(ggplot2)
library(ggraph)
library(magrittr)
library(reshape2)
library(ggpubr)

graphs <- read_dist_to_graphs()

mega_graph <- disjoint_union(graphs)
names <- make.unique(get.vertex.attribute(graph = mega_graph, name = "name"), sep = "_")
mega_graph <- set.vertex.attribute(mega_graph, name =  "name",
                                   value = names)

redlist <- reduce_edges_splitting(mega_graph)
mega_graph_reduced <- delete.edges(mega_graph, which(redlist > 0))



plot_full <-  ggraph(mega_graph, layout = cbind(get.vertex.attribute(mega_graph)$X,
                                                get.vertex.attribute(mega_graph)$Y)) +
  geom_edge_link() + geom_node_point(size = 2, aes(color = Color))
plot_full + ggsave("graph_full.png", type = "cairo", width = 10, height = 15)

plot_reduced <- ggraph(mega_graph_reduced, layout = cbind(get.vertex.attribute(mega_graph)$X,
                                                          -get.vertex.attribute(mega_graph)$Y)) +
  geom_edge_link(color = "white") + geom_node_point(size = 2, aes(color = Color))



plt_reduced_bg <- ggraph(mega_graph_reduced, layout = cbind(get.vertex.attribute(mega_graph)$X,
                                                            -get.vertex.attribute(mega_graph)$Y)) +
  geom_edge_link(color = "white") + geom_node_point(size = 2, aes(color = Color), show.legend = F) + ggpubr::theme_transparent()

plt_reduced_bg + ggsave("graph_to_fit.png", bg = "transparent",  type = "cairo", width = 9.5, height =  21.6)


plot_full <-  ggraph(mega_graph, layout = cbind(get.vertex.attribute(mega_graph)$X,
                                                get.vertex.attribute(mega_graph)$Y)) +
geom_edge_link() + geom_node_point(size = 2, aes(color = Color))
plot_full + ggsave("graph_full.png", type = "cairo", width = 10, height = 15)

plot_reduced <- ggraph(mega_graph_reduced, layout = cbind(get.vertex.attribute(mega_graph)$X,
                                                          -get.vertex.attribute(mega_graph)$Y)) + background_image(img) +
  geom_edge_link(color = "white") + geom_node_point(size = 2, aes(color = Color))
plot_reduced + ggsave("graph_reduced.png", type = "cairo", width = 10, height = 15)






# graphs expansion examples
mygraph <- graphs[[24]]
redlist <- reduce_edges_splitting(mygraph)
mygraph <- delete.edges(mygraph, which(redlist > 0))
expan <- simulate_expansion_of_fraction(mygraph, survivor_fraction = 0.4,expanding_fraction = 0.1)


mplot <- ggraph(mygraph,layout =  cbind(get.vertex.attribute(mygraph,"X"),get.vertex.attribute(mygraph,"Y"))) +
  geom_edge_link()

one <- mplot + geom_node_point(size = 3, aes(color = expan$survivors < 100000 )) + labs(color = "col")
one + ggsave("example1_survivors.png", type = "cairo", width = 6, height = 4)
two <- mplot + geom_node_point(size = 3, aes(color = as.factor(expan$origins))) + labs(color = "col") +  scale_color_brewer(palette =  "Paired")
two + ggsave("example2_origins.png", type = "cairo", width = 6, height = 4)
three <- mplot + geom_node_point(size = 3, aes(color = as.factor(expan$expanded))) + labs(color = "col") +  scale_color_brewer(palette =  "Paired")
three + ggsave("example3_expanded.png", type = "cairo", width = 6, height = 4)

fin_cols <- expan$final
fin_cols[!fin_cols %in% unique(expan$expanded)] <- 100000
four <- mplot +  geom_node_point(size = 3, aes(color = as.factor(fin_cols))) + labs(color = "col") +  scale_color_brewer(palette =  "Paired")
four + ggsave("example4_final_expansion.png", type = "cairo", width = 6, height = 4)
