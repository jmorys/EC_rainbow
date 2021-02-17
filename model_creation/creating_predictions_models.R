setwd("~/studia/zaklad/EC_rainbow/model_creation")
source('~/studia/zaklad/EC_rainbow/model_creation/graph_analysis_tools.R', echo = F)
source('~/studia/zaklad/EC_rainbow/model_creation/fitting_to_each_fragment.R', echo = F)


graph_dir = "D:/EC_rainbow_data/graphs"
files <- list.files(graph_dir)
graphs <- purrr::map(files, function(x){
  graph <- read_rds(paste(graph_dir, x, sep = "/"))
  graph <- as.undirected(graph)
  E(graph)$weights <- E(graph)$weight
  return(graph)
})

output_dir <- "D:/EC_rainbow_data/complete_predictions"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# weird error with NAN in weights (found only 1 edge in sample3?...)
graphs <- modify(graphs, function(graph){
  error_edges <- which(is.na(igraph::get.edge.attribute(graph)$weight))
  graph <- igraph::delete.edges(graph, error_edges)
  graph
})


g_comps <- map(graphs, function(x){
  igraph::decompose(graph = x, min.vertices = 70)
})


names <- str_sub(files, end = -5)

for (i in 1:length(g_comps)) {
  output_dir_graph <- paste(output_dir, names[i], sep = "/")
  if (!dir.exists(output_dir_graph)) {
    dir.create(output_dir_graph)
  }

  for (j in 1:length(g_comps[[i]])) {
    file_name <- paste0(output_dir_graph, "/", "graph", j, ".rds")
    if (file.exists(file_name)) next
    graph <- g_comps[[i]][[j]]
    c_res <- get_complete_results(graph = graph)
    saveRDS(c_res, file = file_name )
  }
}








output_dir <- "D:/EC_rainbow_data/complete_predictions"
global_list <- list()
for (i in 1:length(names)) {
  graph_dir <- paste(output_dir, names[i], sep = "/")
  files <- list.files(graph_dir)
  graphframe <- map_dfr(paste(graph_dir, files, sep = "/"), function(x){
    g_data <- read_rds(x)
    
    c(g_data$graph_data$predictions,size = g_data$graph_data$size)
  })
  global_list[[i]] <- graphframe
}



names(global_list) <- names

global_list <- bind_rows(global_list, .id = "image")


global_list <- global_list %>% mutate(irradiated = !str_detect(image, "non"),
                                      day = str_extract(image,"d[0-9]+"),
                                      image = str_extract(image,"AE[0-9]+"))


global_list %>% ggplot(aes(survival, expansion, shape = irradiated,
                           col = log10(size))) +
  geom_pointrange(aes(ymin = expansion02, ymax = expansion08)) +
  geom_linerange(aes( xmin = survival_02, xmax = survival_08))



survival_plot <- global_list %>% ggplot(aes(x = image, y = survival, color = sqrt(size), fill = irradiated)) + 
  geom_boxplot(aes(weight = sqrt(size) ), size = 1) +
  geom_pointrange(aes( ymin = survival_02, ymax = survival_08),
                  ,position = position_jitter(width = 0.25), size = 1.5, shape = 21, stroke = 2, fatten = 2) +
  facet_wrap(.~day, scales = "free_x") + theme_bw() +
  scale_color_viridis(option = "magma", direction = -1) +
  scale_fill_manual(values = c("grey90", "grey40"))
survival_plot + ggsave("survival_plot.png", type = "cairo", width = 12, height = 8, dpi = 200)

text_colors <- c("grey90", "grey40")[global_list$irradiated + 1]

expansion_plot <- global_list %>% ggplot(aes(x = image, y = expansion, color = sqrt(size), fill = irradiated)) + 
  geom_boxplot(aes(weight = sqrt(size) ), size = 1) +
  geom_pointrange(aes( ymin = expansion02, ymax = expansion08),
                  ,position = position_jitter(width = 0.25), size = 1.5, shape = 21, stroke = 2, fatten = 2) +
  facet_wrap(.~day, scales = "free_x") + theme_bw() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) + scale_color_viridis(option = "magma", direction = -1) +
  scale_fill_manual(values = c("grey90", "grey40"))

expansion_plot + ggsave("expansion_plot.png", type = "cairo", width = 12, height = 8, dpi = 200)















graph <- g_comps[[5]][[1]]
# 
# base_plot <- ggraph(graph, layout = tibble(get.vertex.attribute(graph)$x, get.vertex.attribute(graph)$y)) +
#   geom_edge_link() + scale_y_reverse() + scale_color_discrete()

# base_plot + geom_node_point(size = 2, aes(color = as_factor(colour)), show.legend = F)

a <- Sys.time()
simulation <- simulate_expansion_of_fraction_2(graph, survivor_fraction = 0.7, expanding_fraction = 0.2)

b <- Sys.time()

originz <- !is.na(simulation$origins)
originz <- originz*1.
out <- page.rank(graph, personalized = originz, directed = F, damping = 0.5)
dif1_v <- out$vector
survivors <- !is.na(simulation$survivors)
non_expanding <- (survivors & !originz)
edge_list <- get.edgelist(graph, names = F)
e1s <- dif1_v[edge_list[,1]]
e2s <- dif1_v[edge_list[,2]]
edge_list[e1s < e2s, ] <- cbind(
  edge_list[e1s < e2s,2 ],
  edge_list[e1s < e2s, 1])
edge_params <- tibble(from = edge_list[,1], to = edge_list[,2], potential = dif1_v[ edge_list[,2]])
edge_params <- edge_params %>% filter(from %in% which(originz < 1) & to %in% which(originz < 1))

edge_params <- edge_params %>% arrange(potential)

non_expanding_update <- non_expanding
continue <- T
while (continue) {
  arranged_sub <- edge_params %>% filter(from %in% which(non_expanding_update) & to %in% which(!non_expanding_update) )
  substitution <- arranged_sub %>% group_by(from) %>% filter(row_number() == 1) %>% group_by(to) %>% filter(row_number() == 1)
  non_expanding_update[substitution$from] <- F
  non_expanding_update[substitution$to] <- T
  continue <- dim(substitution)[1] > 0
}
# 
# dir_plot + geom_node_point(size = 4, aes(color = non_expanding_update, shape = originz > 0), show.legend = F) +
#   geom_edge_link(arrow = arrow(angle = 10, length = unit(0.15, "inch"), type = "closed")) + scale_color_discrete()
# 


base_plot <- ggraph(graph, layout = tibble(get.vertex.attribute(graph)$x, get.vertex.attribute(graph)$y)) +
  geom_edge_link() + scale_y_reverse() + scale_color_discrete() 

base_plot + geom_node_point(size = 4, aes(color = non_expanding, shape = originz > 0), show.legend = F)
base_plot + geom_node_point(size = 4, aes(color = non_expanding_update, shape = originz > 0), show.legend = F)




c <- Sys.time()
  








graph_dir = "D:/EC_rainbow_data/graphs"
files <- list.files(graph_dir)
graphs <- purrr::map(files, function(x){
  graph <- read_rds(paste(graph_dir, x, sep = "/"))
  return(graph)
})

data <- purrr::map_dfr(1:length(graphs), function(x){
  edge.attributes(graphs[[x]])
}, .id = "image")


data_e <- data %>% mutate(image = files[as.numeric(image)])  %>% mutate(irradiated = !str_detect(image, "non"),
           day = str_extract(image,"d[0-9]+"),
           image = str_extract(image,"AE[0-9]+"))


distances_plot <- data_e %>% ggplot(aes(image, mean, fill = irradiated)) +
  geom_violin() + facet_wrap(.~day, scales = "free_x") +
  geom_boxplot(width = 0.1) + theme_bw()
distances_plot + ggsave("distances_plot.png", type = "cairo", width = 8, height = 6, dpi = 200)


data <- purrr::map_dfr(1:length(graphs), function(x){
  vertex.attributes(graphs[[x]])
}, .id = "image")


data_v <- data %>% mutate(image = files[as.numeric(image)])  %>% mutate(irradiated = !str_detect(image, "non"),
                                                                        day = str_extract(image,"d[0-9]+"),
                                                                        image = str_extract(image,"AE[0-9]+"))

cols <- c("Cerulean", "mOrange", "mCherry")

color_distribution_plot <- data_v %>% mutate(colour = as_factor(cols[colour])) %>% ggplot(aes(image, fill = colour)) +
  geom_bar(position = "dodge" ) +
  facet_wrap(.~day, scales = "free_x") + theme_bw()
color_distribution_plot + ggsave("color_distribution_plot.png", type = "cairo", width = 8, height = 6, dpi = 200)




# graph24 <- read_rds("graph24.rds")
# mega_graph_reduced <- read_rds("mega_graph_reduced.rds")
# local <- assortativity_local_3(graph =  mega_graph_reduced, val =  get.vertex.attribute(mega_graph_reduced,"Color"), alpha = 0.2)
# local_tib <- as_tibble_col(local)
# 
# r <- (seq(0.1,0.6,by = 0.1))^2
# q_probes <- sort(c(r,0.5,1 - r))
# 
# data_mega <- local_tib %>%  summarise(quantiles = quantile(value, probs = q_probes ),
#                                    quant_range = q_probes, mean = mean(value), sd = sd(value),
#                                    skew = moments::skewness(value), kurt = moments::kurtosis(value)
#                                    )
# 
# data_mega <- data_mega %>% pivot_wider(values_from = quantiles, names_from = quant_range, id_cols = -c(quantiles,quant_range)) %>%
#   ungroup()
# data_mega <- data_mega %>% add_column(expan = 1, surv = 1, Var2 = as_factor(1))
# 
# surv_prediction <- predict(fit_surv, data_mega)
# data_mega_2 <- data_mega %>% bind_cols(surv_prediction)
# expan_prediction <- predict(fit_expan, data_mega_2)
# data_mega_fin <- data_mega %>% bind_cols(surv_p = surv_prediction$.pred, expan_p = expan_prediction$.pred)
# 
# 
# 
# smol_simul <- simulate_expansion_of_fraction(graph24, survivor_fraction = 0.3, expanding_fraction =  0.1)
# c_col <- sample(1:3, length(unique(smol_simul$final)), replace = T)
# 
# smol_size_test <- assortativity_local_3(graph =  graph24, val =  c_col[smol_simul$final], alpha = 0.2)
# local_smol_tib <- as_tibble_col(smol_size_test)
# 
# data_smol <- local_smol_tib %>%  summarise(quantiles = quantile(value, probs = q_probes ),
#                                       quant_range = q_probes, mean = mean(value), sd = sd(value),
#                                       skew = moments::skewness(value), kurt = moments::kurtosis(value)
# )
# 
# data_smol <- data_smol %>% pivot_wider(values_from = quantiles, names_from = quant_range, id_cols = -c(quantiles,quant_range)) %>%
#   ungroup()
# data_smol <- data_smol %>% add_column(expan = 1, surv = 1, Var2 = as_factor(1))
# 
# surv_prediction <- predict(fit_surv, data_smol)
# data_smol_2 <- data_smol %>% bind_cols(surv_prediction)
# expan_prediction <- predict(fit_expan, data_smol_2)
# data_smol_fin <- data_smol %>% bind_cols(surv_p = surv_prediction$.pred, expan_p = expan_prediction$.pred)
# 
# ggraph(graph24, cbind(get.vertex.attribute(graph24)$X, get.vertex.attribute(graph24)$Y)) +
#   geom_node_point(aes(col = as.factor(smol_simul$final)), show.legend = F, size = 5) + geom_edge_link() +
#   ggsave("identities_smol.png", type = "cairo", width = 10, height = 6, dpi = 200)
# 
# ggraph(graph24, cbind(get.vertex.attribute(graph24)$X, get.vertex.attribute(graph24)$Y)) +
#   geom_node_point(aes(col = as.factor(c_col[smol_simul$final])), show.legend = F, size = 5) + geom_edge_link() +
#   ggsave("colors_smol.png", type = "cairo", width = 10, height = 6, dpi = 200)
# 
# ggraph(graph24, cbind(get.vertex.attribute(graph24)$X, get.vertex.attribute(graph24)$Y)) +
#   geom_node_point(aes(col = smol_size_test), size = 5) + geom_edge_link() + scale_color_viridis(option =  "A", direction = -1) +
#   ggsave("assort_local_smol.png", type = "cairo", width = 10, height = 6, dpi = 200)
# 
#  