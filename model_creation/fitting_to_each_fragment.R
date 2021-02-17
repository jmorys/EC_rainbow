library(tidyverse)
library(igraph)
library(ggplot2)
library(ggraph)
library(magrittr)
library(tidymodels)
library(foreach)
library(doParallel)
library(keras)
library(tensorflow)
library(caret)
source('~/studia/zaklad/EC_rainbow/model_creation/graph_analysis_tools.R', echo = F)
tensorflow::tf$random$set_seed(42)

create_interval_model <- function(weights, rate=0.1, in_shape){
  test_dropout <-  layer_dropout(rate = rate)
  input <- layer_input(shape = in_shape)
  
  output <- input %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dense(units = 128,activation = "relu") %>%
    test_dropout(training = T) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
  
  model_interval <- keras_model(input,output)
  # Compile model
  model_interval %>% compile(
    loss = "mse",
    optimizer = optimizer_adam(), metrics = c("mae"))
  set_weights(model_interval, weights = weights)
  model_interval
}


optimize_intervals <- function(weights, predictors, ori_spread, iterations = 100, verbose = F){
  rate <- 0.1
  sigma_diff <- 1
  lr <- 1
  
  in_shape <- dim(predictors)[2]
  
  if (exists("sameness")) {remove("sameness")}
  
  iteration <- 0
  while (sigma_diff > 0.05) {
    iteration <- iteration + 1
    
    model_interval <- create_interval_model(weights, rate, in_shape)
    interval_data <- matrix(nrow = dim(predictors)[1], ncol = iterations)
    
    for (i in 1:iterations)
      interval_data[,i]  <- model_interval %>% predict(predictors)
    
    dat <- apply(interval_data, 1, function(x) x - median(x))
    
    pred_spread <- matrix(dat, nrow = length(dat))
    
    prop <- (mean(abs(pred_spread))/mean(abs(ori_spread)))
    
    if (exists("sameness")) {
      if (sameness != (prop > 1)) {
        lr <- lr*0.6
      }
    }
    sameness <- prop > 1
    rate = rate/(prop^lr)
    if (rate > 0.9) {
      rate <- 0.9
    }
    sigma_diff <- abs(prop - 1)

    if (verbose | (iteration > 100)) {
      print(rate)
      print(paste("sigma_diff", sigma_diff))
    }
    if (iteration > 25) {
      print(paste("Did not converge"))
      break
    }
  }
  return(list(model_interval = model_interval, prop = prop, pred_spread = pred_spread))
}


generate_data_fast <- function(graph, simulation_parameters, alpha =0.7, n_try = 3){
  source('~/studia/zaklad/EC_rainbow/model_creation/graph_analysis_tools.R', echo = F)
  
  fast_outs = list()
  for (i in 1:length(alpha)) {
    fast_outs[[i]] <- fast_assorts_p_1(graph, alpha = alpha[i])
  }
  
  cl <- parallel::makeCluster(7)
  doParallel::registerDoParallel(cl)
  init_size <- length(V(graph))
  date()
  res <- foreach(sim_params = iter(simulation_parameters, by = 'row'), .export = c("graph", "init_size", "fast_outs",
                                                                                   "mix_mat", "simulate_expansion_of_fraction_cheap",
                                                                                   "fast_assorts_p_2"),
                 .packages = c("igraph")) %:% foreach(try = 1:n_try) %dopar% {
                   survivor_f <- sim_params[1]
                   expanding_f <- sim_params[2]
                   set.seed(as.integer(234 + 456789*survivor_f + 123456*expanding_f + try*123))
                   cols <- sample(1:3, init_size, replace = T)
                   test_cheap <- simulate_expansion_of_fraction_cheap(graph = graph,
                                                                      survivor_fraction = survivor_f,
                                                                      expanding_fraction = expanding_f, cols)
                   n_surv <- ceiling(init_size*survivor_f)
                   n_exp <- ceiling(n_surv*expanding_f)
                   
                   asses <- list()
                   for (i in 1:length(fast_outs)) {
                     ass <- fast_assorts_p_2(test_cheap$final, fast_outs[[i]][[1]], fast_outs[[i]][[2]], fast_outs[[i]][[3]])
                     asses[[i]] <- ass
                   }
                   
                   list(asses, c(n_surv, n_exp))
                 }
  date()
  parallel::stopCluster(cl)
  res_ori <- res
  res <- unlist(recursive = F, res)
  return(res)
}


compile_assortativity_data <- function(assortativities_list){
  r <- (seq(0.1, 0.6, by = 0.1))^1.8
  q_probes <- sort(c(r,0.5,1 - r))
  hists <- map(assortativities_list, function(z){
    z <- z[[1]]
    vals <- lapply(z[1:2], function(x){
      mean = mean(x)
      sd = sd(x)
      skew = moments::skewness(x)
      kurt = moments::kurtosis(x)
      quants <- quantile(x, probs = q_probes)
      c1 <- c(mean, sd, skew, kurt)
      names(c1) <- c("mean", "sd", "skew", "kurt")
      c(c1, quants)
    })
    unlist(vals)
  })
  wh_cors <- map_dfr(assortativities_list, function(z){
    z <- z[[1]]
    mat <- matrix(unlist(z[1:2]), nrow = 3)
    mat <- t(mat)
    v_cors <- cor(mat, method = "s")
    vals <- v_cors[upper.tri(v_cors)]
    names(vals) <- 1:length(vals)
    vals
  })
  assortativities_frame <- matrix(unlist(hists), ncol = length(hists))
  assortativities_frame <- t(assortativities_frame)
  colnames(assortativities_frame) <- make.unique(names(hists[[1]]), sep = "_")
  colnames(wh_cors) <- c("one", "two", "three")
  comb_frame <- bind_cols(as_tibble(assortativities_frame), wh_cors)
}


preprocess_simulation <- function(simmulation_data, init_size){
  res <- simmulation_data
  # predict survival successful model !!!!
  params_df <- map_dfr(res, function(x){
    s <- x[[2]][1]
    e <- x[[2]][2]
    list(surv =  s/init_size, pseudo_eras = log2((init_size - s)/e + 1),  expan = e/s )
  })
  # for double assortativity 
  assorts_data <- compile_assortativity_data(res)
  assorts_data <- as.matrix(assorts_data)
  rot_mat <- prcomp(assorts_data, rank. = 13, scale. = T)$rot
  predictors <- assorts_data %*% rot_mat
  comb_frame <- cbind(params_df ,predictors)
  return(list(combined_frame = comb_frame, rot_matrix = rot_mat))
}


create_model <- function(combined_frame, output = "surv") {
  # output - one of "surv", "pseudo_eras", "expan"
  indexes = createDataPartition(1:dim(combined_frame)[1], p = .85, list = F)
  # predictors <- sub_features
  predictors <- as.data.frame(combined_frame %>% select(-surv, -pseudo_eras, -expan))
  predictors <- as.matrix(predictors)
  target <- as.matrix(combined_frame[output] , ncol = 1)
  
  xtrain = predictors[indexes,]
  indexes_val = createDataPartition(1:length(indexes), p = .85, list = F)
  
  xval = xtrain[-indexes_val,]
  xtrain = xtrain[indexes_val,]
  
  xtest = predictors[-indexes,]
  
  # targets <- targets
  ytrain = target[indexes]
  yval = ytrain[-indexes_val]
  ytrain = ytrain[indexes_val]
  ytest = target[-indexes]
  
  model <- keras_model_sequential()
  model %>%
    layer_dense(input_shape = dim(predictors)[2], units = 128, activation = "relu") %>%
    layer_dense(units = 128,activation = "relu") %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(rate = 0.05) %>%
    layer_dense(units = 1, activation = "linear")
  
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_adam(), metrics = c("mae"))
  
  
  history <- model %>% fit(x = xtrain, y = ytrain, epochs = 100, verbose = 0, validation_data = list(xval, yval),
                           callbacks = list(
                             callback_early_stopping(patience = 20, restore_best_weights = T),
                             callback_reduce_lr_on_plateau(factor = 0.3))
  )
  return( list(model = model, history = history, train_n_val_indexes = indexes, train_indexes = indexes_val))
}


as_metrics_df = function(history) {
  
  # create metrics data frame
  df <- as.data.frame(history$metrics)
  
  # pad to epochs if necessary
  pad <- history$params$epochs - nrow(df)
  pad_data <- list()
  for (metric in history$params$metrics)
    pad_data[[metric]] <- rep_len(NA, pad)
  df <- rbind(df, pad_data)
  
  # return df
  df
}


evaluate_model <- function(combined_frame, c_model_res, output = "surv"){
  history <- as_metrics_df(c_model_res$history)

  history$epoch <- 1:dim(history)[1]
  history <- history %>% pivot_longer(cols = !epoch, names_to = "names")
  history <- history %>% mutate(validation = str_detect(names, ".+_"), names = str_extract(names, "(?!.+_)[^_]+") )

  history_plot <- history %>% ggplot(aes(epoch, value, col = validation)) + geom_line(size = 1) +
    facet_grid(names~., scales = "free_y") + ylim(c(0,NA))
  
  
  train_index <- 1:dim(combined_frame)[1] %in% c_model_res$train_n_val_indexes[c_model_res$train_indexes] 
  predictors <- as.data.frame(combined_frame %>% select(-surv, -pseudo_eras, -expan))
  predictors <- as.matrix(predictors)
  target <- combined_frame[, output]

  xtest <- predictors[-c_model_res$train_n_val_indexes,]
  ytest <- target[-c_model_res$train_n_val_indexes]
  scores = c_model_res$model %>% evaluate(xtest, ytest, verbose = 0)
  pred <- c_model_res$model %>% predict(xtest, verbose = 0)
  
  metrics <- c(scores, (cor(ytest, pred))^2)
  names(metrics) <- c("mse", "mae", "rsq")
  
  
  
  whole_ds_pred <- c_model_res$model %>% predict(predictors)
  plot_d <- bind_cols(simulated = target, predicted = whole_ds_pred[,1], group = train_index)
  reg_plot <- plot_d %>%
    ggplot(aes(simulated, predicted)) + geom_point(aes(col = group) ,alpha = 0.15, shape = 16 ) +
    geom_smooth(aes(group = as.factor(group))) + labs(color = "Used for\ntraining") + 
    geom_abline(slope = 1, col = "red", size = 1)
  error_plot <- plot_d %>% ggplot(aes(  (simulated - predicted))) +
    geom_histogram(aes(fill = group, y = after_stat(ncount)), position = "identity", alpha = 0.5) +
    labs(fill = "Used for\ntraining")
  
  return(list(metrics = metrics, predictions = plot_d,
              regression_plot = reg_plot, error_plot = error_plot, history_plot = history_plot))
}


calculate_intervals <- function(predictors, model, interval = 0.6, iterations = 200){
  iterations <- 200
  interval_ori <- matrix(nrow = dim(predictors)[1], ncol = iterations)
  for (i in 1:iterations)
    interval_ori[,i]  <- model %>% predict(predictors)
  
  confidence_interval <- 0.6
  quantile(interval_ori, c(0.5 - confidence_interval/2, 0.5 + confidence_interval/2) )
}



# res7 <- generate_data_fast(graph, survivor_fraction, expanding_fraction, alpha = 0.7, n_try = 3)
# res3 <- generate_data_fast(graph, survivor_fraction, expanding_fraction, alpha = 0.3, n_try = 3)
# res <- large_res_04_07_095
# res <- large_res_04_07



# smal_res_04_07


get_complete_results <- function(graph){
  set.seed(42)
  survivor_fraction <- runif(4000, 0.6, 0.98)
  set.seed(42)
  expanding_fraction <-  runif(4000, 0.005, 0.6)
  sim_params <- cbind(survivor_fraction, expanding_fraction)
  set.seed(123)
  res_surv <- generate_data_fast(graph, sim_params, alpha = c(0.4, 0.7), n_try = 1)
  init_size <- length(V(graph))
  surv_data <- preprocess_simulation(res_surv, init_size)
  
  
  ass_ori_04 <- assortativity_local_par(graph, get.vertex.attribute(graph)$colour, alpha = 0.4)
  ass_ori_07 <- assortativity_local_par(graph, get.vertex.attribute(graph)$colour, alpha = 0.7)
  ass_ori_data <- list(list(list(ass_ori_04, ass_ori_07)))
  assorts_data_ori <- compile_assortativity_data(ass_ori_data)
  assorts_data_ori <- as.matrix(assorts_data_ori)
  
  
  
  model_surv_d <- create_model(surv_data$combined_frame)
  eval_surv <- evaluate_model(surv_data$combined_frame, model_surv_d)
  surv_ori_spread <- eval_surv$predictions$predicted - eval_surv$predictions$simulated
  optimization_surv <- optimize_intervals(get_weights(model_surv_d$model),
                                          as.matrix(surv_data$combined_frame %>% select(-surv, -pseudo_eras, -expan)),
                                          ori_spread = surv_ori_spread,
                                          iterations = 250)
  
  # pred_spread <-  as.vector(optimize_data[[3]])
  # origin <- c(rep(T, length(ori_spread)),  rep(F, length(pred_spread)))
  # spread <- c(ori_spread, as.vector(pred_spread))
  # spread <- tibble(diff = spread, origin = origin)
  # spread %>% ggplot(aes(diff, fill = origin)) +
  #   geom_histogram(aes(y = after_stat(ncount)), position = "identity", alpha = 0.6)
  predictors_ori_surv <- assorts_data_ori %*% surv_data$rot_matrix
  survival_prediction <- model_surv_d$model %>% predict(predictors_ori_surv)
  
  
  survivor_fraction <- round(survival_prediction * init_size, 0) / init_size
  survivor_fraction <- rep(survivor_fraction, 3000)
  set.seed(42)
  expanding_fraction <-  runif(3000, 1/init_size, 0.7)
  sim_expansion <- cbind(survivor_fraction, expanding_fraction)
  set.seed(42)
  res_expansion <- generate_data_fast(graph, sim_expansion, alpha = c(0.4, 0.7), n_try = 1)
  
  expan_data <- preprocess_simulation(res_expansion, init_size)
  
  model_expan_d <- create_model(expan_data$combined_frame, output = "expan")
  eval_expan <- evaluate_model(expan_data$combined_frame, model_expan_d, output = "expan")
  expan_ori_spread <- eval_expan$predictions$predicted - eval_expan$predictions$simulated
  
  optimization_expan <- optimize_intervals(get_weights(model_expan_d$model),
                                           as.matrix(expan_data$combined_frame %>% select(-surv, -pseudo_eras, -expan)),
                                           ori_spread = expan_ori_spread,
                                           iterations = 250)
  
  predictors_ori_expan <- assorts_data_ori %*% expan_data$rot_matrix
  expan_prediction <- model_expan_d$model %>% predict(predictors_ori_expan)
  
  intervals_surv =  calculate_intervals(predictors = predictors_ori_surv, optimization_surv$model_interval)
  intervals_expan = calculate_intervals(predictors = predictors_ori_expan, optimization_expan$model_interval)
  combined_predictions <-  c(survival_prediction, expan_prediction, intervals_surv, intervals_expan)
  names(combined_predictions) <- c("survival", "expansion", "survival_02", "survival_08",
                                   "expansion02", "expansion08")
  
  complete_results <- list(survival = list(res = res_surv, data = surv_data, model_data = model_surv_d,
                                           evaluation = eval_surv, interval = optimization_surv),
                           expansion = list(res = res_expansion, data = expan_data, model_data = model_expan_d,
                                            evaluation = eval_expan, interval = optimization_expan),
                           graph_data = list(assortativities = list(ass_ori_04, ass_ori_07), data = ass_ori_data,
                                             # predictions 
                                             predictions = combined_predictions, size = init_size)
  )
  return(complete_results)
}






# c_res <- get_complete_results(graph = graph)

# model_interval <- results[[1]]
# test_predicts <- predictors[1:100, ]
# 
# iterations <- 100
# interval_data <- matrix(nrow = dim(test_predicts)[1], ncol = iterations)
# for (i in 1:iterations)
#   interval_data[,i]  <- model_interval %>% predict(test_predicts)
# 
# 
# confidence_interval <- 0.6
# 
# calc_conf_interval <- apply(interval_data, 1, function(x){
#   quantile(x, c(0.5 - confidence_interval/2, 0.5 + confidence_interval/2 ))
# })
# calc_conf_interval <- t(calc_conf_interval)
# 
# honest_pred <- model %>% predict(test_predicts)
# target_test <- target[1:100]
# testing_data <- tibble(true = target_test, pred = honest_pred,
#                        yp = calc_conf_interval[,1], ym = calc_conf_interval[,2] )
# 
# 
# testing_data %>% ggplot(aes(true, pred)) + geom_point() +
#   geom_pointrange(aes(ymin = yp, ymax = ym)) + geom_abline(slope = 1, col = "red", size = 2)




# 
# 
# 
# 
# ggraph(graph, layout = tibble(get.vertex.attribute(graph)$x, get.vertex.attribute(graph)$y)) +
#   geom_edge_link() + geom_node_point(size = 2, aes(color = as_factor(colour)), show.legend = F) + scale_y_reverse()

# 




# 
# 
# d_true <- density(ass_ori_07, bw =  0.01)$y
# 
# 
# bhattach_dist <- unlist(lapply(res, function(x){
#   d_try <- density(x[[1]][[2]],  bw =  0.01)$y
#   -log(sum(sqrt(d_true * d_try)))
# }))
# 
# 
# bhatt_tib <- params_df %>% add_column(dist = bhattach_dist)
# 
# bhatt_tib %>% ggplot(aes(surv, expan, col = dist)) + geom_point(size = 5, alpha = 0.6, shape = 16) +
#   scale_color_viridis_c(option = "magma")
# 
# bhatt_tib %>% ggplot(aes(surv, dist)) + geom_point(size = 2, alpha = 0.2, shape = 16) + geom_smooth()
# 
# 


# 
# 
# 
# d_true <- density(ass_ori_07, bw =  0.005)$y
# bhattach_dist <- unlist(lapply(res, function(x){
#   d_try <- density(x[[1]][[2]],  bw =  0.005)$y
#   -log(sum(sqrt(d_true * d_try)))
# }))
# 
# 
# bhatt_tib <- params_df_expan %>% add_column(dist = bhattach_dist)
# # 
# # bhatt_tib %>% ggplot(aes(surv, expan, col = dist)) + geom_point(size = 5, alpha = 0.6, shape = 16) +
# #   scale_color_viridis_c(option = "magma")
# 
# bhatt_tib %>% ggplot(aes(expan, dist)) + geom_point(size = 2, alpha = 0.2, shape = 16) + geom_smooth()
# 
