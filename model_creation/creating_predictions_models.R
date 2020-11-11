setwd("~/studia/zaklad/EC_rainbow/model_creation")
source('~/studia/zaklad/EC_rainbow/model_creation/graph_analysis_tools.R', echo = TRUE)
rangee <- seq(0.0001,1,length.out = 30)^1.7



graph <- read_rds("graph53.rds")


assortativities <- list()
cores = detectCores()
cl <- makeCluster(3)
date()
pb <- txtProgressBar(style = 3, max = length(rangee)^2)
setTxtProgressBar(pb, 0)
for (k in 1:length(rangee)) {
  assortativities_sub <- list()
  for (j in 1:length(rangee)) {
    registerDoParallel(cl)
    fraction_matrix <- list()
    for (f in 1:3) {
      fraction_matrix[[f]] <- foreach(i = 1:3, .combine = cbind, .verbose = F, .packages = c("data.table", "igraph","tibble","dplyr")) %dopar% {
        library(igraph)
        library(tibble)
        library(dplyr)
        expp <- simulate_expansion_of_fraction(graph, survivor_fraction = rangee[j], expanding_fraction = rangee[k])$final
        set.seed(1000*k + 100*j + 10*f + i)
        cols <- sample(1:3, length(unique(expp)), replace = T)
        tempMatrix = assortativity_local_3(graph, cols[expp], alpha = 0.3)
      }
    }
    assortativities_sub[[j]] <- cbind(fraction_matrix[[1]], fraction_matrix[[2]])
    setTxtProgressBar(pb, (k - 1)*length(rangee) + j)
  }
  assortativities[[k]] <- assortativities_sub
}

date()
write_rds(assortativities, "assortativities_for_model.rds")
assortativities_original <- read_rds("assortativities_for_model.rds")

join_list <- list()
for (j in 1:length(assortativities_original)) {
  assortativities <- assortativities_original[[j]]
  
  melt_list <- lapply(assortativities, function(x){
    colnames(x) <- paste0("result.",1:dim(x)[2])
    melt(x)
  })
  for (i in 1:length(rangee)) {
    melt_list[[i]][,4] <- rangee[i]
  }
  joined_melt <- as_tibble(bind_rows(melt_list, .id = "surv"))
  joined_melt$surv <- joined_melt$V4
  join_list[[j]] <- joined_melt
}

turbojoin <- as_tibble(bind_rows(join_list, .id = "expan"))
turbojoin$expan <- rangee[as.numeric(turbojoin$expan)]
nlen <- length(V(graph))
turbojoin <- turbojoin %>% mutate(true_surv = ceiling(surv * nlen)/nlen)
turbojoin <- turbojoin %>% mutate(true_expan = ceiling(true_surv * nlen*expan)/(nlen * true_surv))

melt_summary <- turbojoin %>% group_by(surv, expan , Var2) %>% summarise(mean = mean(value), sd = sd(value),
                                                                         skew = moments::skewness(value), kurt = moments::kurtosis(value))
sum_summary <- melt_summary %>% group_by(surv, expan ) %>% summarise(mean = mean(mean), sd = mean(sd))



trueexpan_plt <- ggplot(melt_summary, aes(expan*surv, mean, col = as_factor(round(surv,8)))) + geom_point() +
  geom_smooth(span = 0.3, level = 0.8)

# absolute expanding fraction ~ mean assortativity  
trueexpan_plt + ggsave("trueexpan_plt.png", type = "cairo", width = 10, height = 6, dpi = 200)

pseudo_eras <- ggplot(data, aes(log2((1 - true_surv)/(true_surv*true_expan) + 1), skew, col = as_factor(round(surv,8)) ) ) +
  geom_point(show.legend = F) +
  geom_smooth( level = 0.8) + facet_wrap(~as_factor(round(surv,8)), scales = "free" )
#  pseudo(val+1) logarithm of nonliving cells/absolute expanding fraction  ~ skewness
pseudo_eras
pseudo_eras + ggsave("pseudo_eras_plt.png", type = "cairo",  width = 16, height = 12, dpi = 150)


# model creation
r <- (seq(0.1,0.6,by = 0.1))^2
q_probes <- sort(c(r,0.5,1 - r))

test_q <- turbojoin %>% group_by(expan, surv, Var2, true_surv, true_expan) %>%
  summarise(quantiles = quantile(value, probs = q_probes ),
            quant_range = q_probes, mean = mean(value), sd = sd(value),
            skew = moments::skewness(value), kurt = moments::kurtosis(value)
            )

data <- test_q %>% pivot_wider(values_from = quantiles, names_from = quant_range, id_cols = -c(quantiles,quant_range)) %>% ungroup()


data_df <- data %>% select(-true_expan)
set.seed(123)
data_split <- initial_split(data_df, prop = 0.7)
data_train <- training(data_split)
data_test <- testing(data_split)


recipe_surv <-  recipe(true_surv~. , data_train) %>% update_role(expan, surv, Var2, new_role = "ID") %>% 
  step_zv(all_predictors())

tuning_surv <- .GlobalEnv$two_step_xgb_tune(recipe = recipe_surv, data_train = data_train)

final_xgb_surv <- tuning_surv$final_xgb

final_res_surv <- last_fit(final_xgb_surv, data_split)
collect_metrics(final_res_surv)
collect_predictions(final_res_surv) %>% ggplot(aes(true_surv, .pred)) + geom_abline(slope = 1, color = "red3", size = 1) +
  geom_point() 

fit_surv <- final_xgb_surv %>%
  fit(data_train)
saveRDS(fit_surv, "fit_surv.rds")
surv_prediction <- predict(fit_surv, data)






data_df <- data %>% bind_cols(surv_prediction) %>% select(-true_surv)
set.seed(123)
data_split <- initial_split(data_df, prop = 0.7)
data_train <- training(data_split)
data_test <- testing(data_split)


recipe_expan <-  recipe(true_expan~. , data_train) %>% update_role(expan, surv, Var2, new_role = "ID") %>% 
  step_zv(all_predictors())


tuning_expan <- .GlobalEnv$two_step_xgb_tune(recipe = recipe_expan, data_train = data_train)

final_xgb_expan <- tuning_expan$final_xgb



final_res_expan <- last_fit(final_xgb_expan, data_split)
collect_metrics(final_res_expan)
collect_predictions(final_res_expan) %>% ggplot(aes(true_expan, .pred)) + geom_abline(slope = 1, color = "red3", size = 1) +
  geom_point() 

fit_expan <- final_xgb_expan %>%
  fit(data_train)
saveRDS(fit_expan, "fit_expan.rds")

expan_prediction <- predict(fit_expan, data_df)

# most important predictors
params_import_expan <- fit_expan %>%
  pull_workflow_fit() %>%
  vip::vip(geom = "point")
params_import_expan + ggsave("params_import_expan.png", type = "cairo", width = 6, height = 4)

plot_data <- data %>% bind_cols(surv_p = surv_prediction$.pred, expan_p = expan_prediction$.pred)

expan_true_pred_plt <- ggplot(plot_data, aes(true_expan, expan_p, col = as_factor(round(surv, 8)))) + geom_point(show.legend = F) +
  geom_smooth(show.legend = F) + facet_wrap(~as_factor(round(surv, 8)))
# expanding fraction fit quality split by surviving fraction
expan_true_pred_plt + ggsave("expan_true_pred_plt.png", type = "cairo", width = 16, height = 12, dpi = 150)

surv_true_pred_plt <- ggplot(plot_data, aes(true_surv, surv_p, col = as_factor(round(expan, 8)) )) + geom_point() + geom_smooth()
# surviving fraction fit quality split by expanding fraction
surv_true_pred_plt + ggsave("surv_true_pred_plt.png", type = "cairo", width = 10, height = 6, dpi = 200)


expansd_surv_log_plt <- ggplot(plot_data, aes(surv_p, log10(abs(expan_p/true_expan)), col = as_factor(round(expan, 8)))) + geom_point()
# log10 absolute residuals
expansd_surv_log_plt + ggsave("expansd_surv_log_plt.png", type = "cairo", width = 10, height = 6, dpi = 200)


expansd_surv_plt <- ggplot(plot_data, aes(surv_p, expan_p - true_expan, col = as_factor(round(expan, 8)))) +
  geom_point() + geom_smooth(inherit.aes = F,aes(surv_p, expan_p - true_expan), level = 0.999, span = 0.1)
# residuals
expansd_surv_plt + ggsave("expansd_surv_plt.png", type = "cairo", width = 10, height = 6, dpi = 200)








graph24 <- read_rds("graph24.rds")
mega_graph_reduced <- read_rds("mega_graph_reduced.rds")
local <- assortativity_local_3(graph =  mega_graph_reduced, val =  get.vertex.attribute(mega_graph_reduced,"Color"), alpha = 0.2)
local_tib <- as_tibble_col(local)

r <- (seq(0.1,0.6,by = 0.1))^2
q_probes <- sort(c(r,0.5,1 - r))

data_mega <- local_tib %>%  summarise(quantiles = quantile(value, probs = q_probes ),
                                   quant_range = q_probes, mean = mean(value), sd = sd(value),
                                   skew = moments::skewness(value), kurt = moments::kurtosis(value)
                                   )

data_mega <- data_mega %>% pivot_wider(values_from = quantiles, names_from = quant_range, id_cols = -c(quantiles,quant_range)) %>%
  ungroup()
data_mega <- data_mega %>% add_column(expan = 1, surv = 1, Var2 = as_factor(1))

surv_prediction <- predict(fit_surv, data_mega)
data_mega_2 <- data_mega %>% bind_cols(surv_prediction)
expan_prediction <- predict(fit_expan, data_mega_2)
data_mega_fin <- data_mega %>% bind_cols(surv_p = surv_prediction$.pred, expan_p = expan_prediction$.pred)



smol_simul <- simulate_expansion_of_fraction(graph24, survivor_fraction = 0.3, expanding_fraction =  0.1)
c_col <- sample(1:3, length(unique(smol_simul$final)), replace = T)

smol_size_test <- assortativity_local_3(graph =  graph24, val =  c_col[smol_simul$final], alpha = 0.2)
local_smol_tib <- as_tibble_col(smol_size_test)

data_smol <- local_smol_tib %>%  summarise(quantiles = quantile(value, probs = q_probes ),
                                      quant_range = q_probes, mean = mean(value), sd = sd(value),
                                      skew = moments::skewness(value), kurt = moments::kurtosis(value)
)

data_smol <- data_smol %>% pivot_wider(values_from = quantiles, names_from = quant_range, id_cols = -c(quantiles,quant_range)) %>%
  ungroup()
data_smol <- data_smol %>% add_column(expan = 1, surv = 1, Var2 = as_factor(1))

surv_prediction <- predict(fit_surv, data_smol)
data_smol_2 <- data_smol %>% bind_cols(surv_prediction)
expan_prediction <- predict(fit_expan, data_smol_2)
data_smol_fin <- data_smol %>% bind_cols(surv_p = surv_prediction$.pred, expan_p = expan_prediction$.pred)

ggraph(graph24, cbind(get.vertex.attribute(graph24)$X, get.vertex.attribute(graph24)$Y)) +
  geom_node_point(aes(col = as.factor(smol_simul$final)), show.legend = F, size = 5) + geom_edge_link() +
  ggsave("identities_smol.png", type = "cairo", width = 10, height = 6, dpi = 200)

ggraph(graph24, cbind(get.vertex.attribute(graph24)$X, get.vertex.attribute(graph24)$Y)) +
  geom_node_point(aes(col = as.factor(c_col[smol_simul$final])), show.legend = F, size = 5) + geom_edge_link() +
  ggsave("colors_smol.png", type = "cairo", width = 10, height = 6, dpi = 200)

ggraph(graph24, cbind(get.vertex.attribute(graph24)$X, get.vertex.attribute(graph24)$Y)) +
  geom_node_point(aes(col = smol_size_test), size = 5) + geom_edge_link() + scale_color_viridis(option =  "A", direction = -1) +
  ggsave("assort_local_smol.png", type = "cairo", width = 10, height = 6, dpi = 200)

 