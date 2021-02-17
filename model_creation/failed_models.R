

# 
# 
# valz <-  params_df %>% add_column(dist = bhattach_dist)
# 
# 
# valz %>% ggplot(aes(expan, dist, col = surv)) + geom_point() +
# geom_smooth(aes(group = surv), level = 0.99) + scale_color_viridis(option = "magma")
# 
# 
# valz %>% ggplot(aes(pseudo_eras, dist, col = surv)) + geom_point() +
# geom_smooth(aes(group = surv), level = 0.99) + scale_color_viridis(option = "magma")
# 
# 
# #
# valz2 <- tibble(surv = rep(survivor_fraction, each = length(expanding_fraction)*3),
#                 expan = rep(expanding_fraction, length(survivor_fraction), each = 3), dist = bhattach_dist)
# 
# valz2 <- valz2 %>% dplyr::group_by(surv, expan) %>% dplyr::summarise(dist = mean(dist))
# 
# valz2 %>% ggplot(aes(surv, expan)) + geom_raster(aes(fill = dist), interpolate = T) + scale_fill_viridis(option = "magma")
# 
# 

#



#
#

# hyper_params <- map(final_xgb_expan$fit$actions$model$spec$args[1:7], function(x){x[2]})

distribution_pooled <- map_dfr(res, function(x){
  x <- x[[1]]
  list(x = x[[1]], y = x[[2]], z = x[[3]])
})


distr_pca <- prcomp(distribution_pooled, scale. = T, rank. = 2)
pca_rot <- distr_pca$rotation



samp <- res[[2]][[1]]
distribution <-  tibble(x = samp[[1]], y = samp[[2]], z = samp[[3]])
# 
distribution %>% ggplot(aes(x, z)) + stat_density_2d(
  geom = "raster", aes(fill = after_stat(density)),
  contour = FALSE, adjust = 0.3, interpolate = F) + xlim(c(-1,1)) + ylim(c(-1,1)) +
  geom_quantile(quantiles = q_probes, method = "rqss", lambda = 0.5) + scale_fill_viridis_c(option = "magma")


# 
# uniprobes <- seq(0,1, length.out = 10)
# 
# quants_95 <- quantile(distribution$z, uniprobes)
# 
# levs <- cut(distribution$z, quants_95)
# levs <- as.numeric(levs)
# dist_q_ext <- distribution %>% add_column(levs)
# dist_q_ext <- dist_q_ext  %>% group_by(levs) %>% summarise_all( quantile,uniprobes)
# 
# dist_q_ext_mat <- matrix(dist_q_ext$x, ncol = 10)
# heatmap(x = dist_q_ext_mat, scale = "none", Colv = NA, Rowv = NA)
# 
# 

distribution %>% ggplot(aes(x, z)) + stat_bin2d(
  geom = "raster", aes(fill = after_stat(density)), interpolate = F, bins = 30) + xlim(c(-1,1)) + ylim(c(-1,1)) +
  scale_fill_viridis_c(option = "magma")

dens_2d <- MASS::kde2d( distribution$z,distribution$x,lims = c(-1,1,-1,1), h = c(0.1,0.1))
heatmap(x = dens_2d[[3]], scale = "none", Colv = NA, Rowv = NA)



dist_pca <- as_tibble(as.matrix(distribution) %*% pca_rot)
quants_pc1 <- quantile(dist_pca$PC1, c(0, q_probes, 1))
quants_pc2 <- quantile(dist_pca$PC2,  c(0, q_probes, 1))

levs <- cut(dist_pca$PC1, quants_pc1)
levs <- as.numeric(levs)
dist_pcaext <- dist_pca %>% add_column(levs)
dist_pcaext <- dist_pcaext  %>% group_by(levs) %>% summarise_all( quantile,c(0.25,0.75))

dist_pca %>% ggplot(aes(PC1, PC2)) + geom_density_2d_filled(adjust = 0.4) +
  geom_vline(xintercept = quants_pc1, col = "orange") +   geom_hline(yintercept = quants_pc2, col = "orange") + 
  geom_point(data =  dist_pcaext, aes(PC1, PC2), col = "red") + xlim(c(-0.8, 1.6)) + ylim(c(-0.6, 0.6))




whi85 <- which(round(params_df$surv, 2) == 0.85)
whi85 
samp <- res[rev(whi85)[1:10]]
# samp <- res[7000:7010]

distr_pca <- map_dfr(1:10, function(i){
  x <- samp[[i]]
  x1 <- x[[1]]
  distribution <-  tibble(x = x1[[1]], y = x1[[2]], z = x1[[3]])
  distr_pca <- as.matrix(distribution) %*% pca_rot
  
  distr_pca <- as_tibble(distr_pca) %>% add_column(iter = i)
  distr_pca
})


# distr_pca %>% ggplot(aes(PC1,PC2)) + geom_density_2d_filled(adjust = 0.5) + xlim(c(-1,1)) + ylim(c(-1,1))

# 
# distr_pca %>% ggplot(aes(PC1,PC2)) + stat_density_2d(
#   geom = "raster", aes(fill = after_stat(density)),
#   contour = FALSE, adjust = 0.3, interpolate = T) + scale_fill_viridis_c() + 
#   xlim(c(-1,1)) + ylim(c(-1,1)) + facet_wrap(~iter)
# 


# distr_pca %>% ggplot(aes(PC1,PC2)) + geom_point(alpha = 0.7, aes(col = as_factor(iter)))



# super cool plot
ggplot(distr_pca, aes(x = .panel_x, y = .panel_y) ) + 
  geom_autodensity(position = "identity", alpha = 0.15, size = 1, aes(group = as_factor(iter), col = as_factor(iter)) ) +
  geom_density2d_filled(contour_var = "ndensity" , adjust = 0.5) +
  geom_point(alpha = 0.4, shape = 16, size = 1.5, aes(col = as_factor(iter))) + 
  facet_matrix(vars(-iter), layer.diag = 1, layer.upper = 2)






xgb_specs_surv <- boost_tree(
  trees = 800,
  tree_depth = tune(), min_n = 15,
  loss_reduction = tune(),                     ## first three: model complexity
  sample_size = 0.4, mtry = tune(),         ## randomness
  learn_rate = tune(),                         ## step size
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")


r <- (seq(0.1,0.6,by = 0.1))^2
q_probes <- sort(c(r,0.5,1 - r))


# x <- samp[[i]]
# x1 <- x[[1]]
# distribution <-  tibble(x = x1[[1]], y = x1[[2]], z = x1[[3]])
# distr_pca <- as.matrix(distribution) %*% pca_rot
# 


hists <- map(res, function(z){
  z <- z[[1]]
  x <- z[[2]]
  mean = mean(x)
  sd = sd(x)
  skew = moments::skewness(x)
  kurt = moments::kurtosis(x)
  quants <- quantile(x, probs = q_probes)
  c1 <- c(mean, sd, skew, kurt)
  names(c1) <- c("mean", "sd", "skew", "kurt")
  
  levs <- cut(x, quants)
  levs <- as.numeric(levs)
  distribution <- tibble(x = z[[3]])
  distribution <- distribution %>% add_column(levs)
  distribution <- distribution  %>% group_by(levs) %>% summarise_all(median)
  
  c(c1, quants, distribution$x)
})



hists <- map(res, function(z){
  z <- z[[1]]
  distribution <-  tibble(x = z[[1]], y = z[[2]], z = z[[3]])
  apply(distribution, 2, function(x){
    mean = mean(x)
    sd = sd(x)
    skew = moments::skewness(x)
    kurt = moments::kurtosis(x)
    quants <- quantile(x, probs = q_probes)
    c1 <- c(mean, sd, skew, kurt)
    names(c1) <- c("mean", "sd", "skew", "kurt")
    c(c1, quants)
  })
})




hists <- map_dfr(res, function(z){
  z <- z[[1]]
  distribution <-  tibble(x = z[[1]], y = z[[2]], z = z[[3]])
  x <- distribution$z
  mean = mean(x)
  sd = sd(x)
  skew = moments::skewness(x)
  kurt = moments::kurtosis(x)
  quants <- quantile(x, probs = q_probes)
  c1 <- c(mean, sd, skew, kurt)
  names(c1) <- c("mean", "sd", "skew", "kurt")
  c(c1, quants)
  
})
assortativities_frame <- hists
# assortativities_frame = matrix(unlist(hists), c(length(hists), length(hists[[1]])) )

# c_names <- unlist(lapply(as.character(c(0.5, 0.7, 0.9)),
# function(x) paste(x, c("mean", "sd", "skew", "kurt", as.character(q_probes)), sep = "_")))
# c_names <- c("mean", "sd", "skew", "kurt", as.character(q_probes),paste("deriv",as.character(q_probes)))
# 
# colnames(assortativities_frame) <- c_names
# assortativities_frame <- as_tibble(assortativities_frame)

# assortativities_frame1 <- assortativities_frame
# assortativities_frame2 <- assortativities_frame


comb_frame <- bind_cols(params_df, assortativities_frame)





data_df <- comb_frame %>% select(-pseudo_eras, -expan, 
                                 # -starts_with("0.5_")
)

set.seed(123)
data_split <- initial_split(data_df, prop = 0.7)
data_train <- training(data_split)
data_test <- testing(data_split)


recipe_surv <-  recipe(surv~. ,data_train) %>%
  step_zv(all_predictors())

workflow_surv <- workflow() %>%
  add_recipe(recipe_surv) %>%
  add_model(xgb_specs_surv)

data_folds <- bootstraps(data_df, times = 10)



xgb_set <- parameters(workflow_surv)
xgb_set <- xgb_set %>%
  update(tree_depth = num_comp(c(5, 30)),
         learn_rate = learn_rate(),
         loss_reduction  =  loss_reduction(),
         mtry = num_comp(c(1, 25)))




doParallel::registerDoParallel(cores = parallel::detectCores() - 1)
set.seed(234)
search_res <-
  workflow_surv %>%
  tune_bayes(
    resamples = data_folds,
    param_info = xgb_set,
    initial = 7,
    iter = 10,
    control = control_bayes(no_improve = 10, verbose = TRUE)
  )
estimates <-
  collect_metrics(search_res) %>%
  arrange(.iter)

show_best(search_res, metric = "rsq")
autoplot(search_res, type = "performance")


best_rmse <- select_best(search_res)
workflow_surv_fin <- finalize_workflow(workflow_surv, best_rmse)
workflow_surv_fin_fit <- fit(workflow_surv_fin, data_train)


comb_frame %>% add_column(pred =  predict(workflow_surv_fin_fit, data_df)$.pred) %>%
  ggplot(aes(surv, pred, col = expan)) + geom_abline(slope = 1, color = "red3", size = 1) +
  geom_point(alpha = 0.4) + scale_color_viridis(option = "magma") + geom_smooth(level = 0.999, span = 0.2) + 
  geom_quantile(color="red")

comb_frame %>% add_column(pred =  predict(workflow_surv_fin_fit, data_df)$.pred) %>%
  ggplot(aes(surv, (pred - surv)^2, col = expan)) + geom_point() + geom_smooth(level = 0.999)


comb_frame %>% add_column(pred =  predict(workflow_surv_fin_fit, data_df)$.pred) %>%
  ggplot(aes(pred - surv)) + geom_histogram()

data_train %>% add_column(pred =  predict(workflow_surv_fin_fit, data_train)$.pred) %>%
  ggplot(aes(pred - surv)) + geom_histogram()
data_test %>% add_column(pred =  predict(workflow_surv_fin_fit, data_test)$.pred) %>%
  ggplot(aes(pred - surv)) + geom_histogram()



last_rf_fit <- 
  workflow_surv_fin %>% 
  last_fit(data_split)

last_rf_fit %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip(num_features = 10)





ori_ass_frame <- map_dfr(list(ass_ori), function(x){
  mean = mean(x)
  sd = sd(x)
  skew = moments::skewness(x)
  kurt = moments::kurtosis(x)
  qunts <- quantile(x, probs = q_probes)
  c1 <- c(mean, sd, skew, kurt)
  names(c1) <- c("mean", "sd", "skew", "kurt")
  c(c1, qunts)
})
predict(workflow_surv_fin_fit, ori_ass_frame)$.pred















library(keras)
library(tensorflow)
library(caret)


# #heatmaps approach - fail
# whole_heatmaps <- lapply(res, function(x){
#   x1 <- x[[1]]
#   distribution <-  tibble(x = x1[[1]], y = x1[[2]], z = x1[[3]])
#   dens_2d <- MASS::kde2d( distribution$y,distribution$x,lims = c(-1,1,-1,1), h = c(0.08,0.08), n = 100)
#   dens_2d$z
# })


# whole_heatmaps_flat <- sapply(whole_heatmaps, function(x){
#   matrix(x, nrow = 1)
# })
# whole_heatmaps_means <- rowMeans(whole_heatmaps_flat)
# whole_heatmaps_sds <- apply(whole_heatmaps_flat, 1, sd) 
# me_sd <- tibble(mean = whole_heatmaps_means, sd = whole_heatmaps_sds)
# 
# me_sd %>% ggplot(aes(mean, sd)) + geom_point()
# me_sd %>% ggplot(aes(mean)) + geom_histogram(bins = 50)
# me_sd %>% ggplot(aes(sd)) + geom_histogram(bins = 50)

# trash <- (me_sd$mean < 1e-6 & me_sd$sd < 1e-6)
# trash_m <- matrix(trash_m, ncol = 25)
# heatmap(x = trash_m*1, scale = "none", Colv = NA, Rowv = NA)
# 
# 
# 
# sub_features <- whole_heatmaps_flat[!trash,]
# sub_features <- t(sub_features)

# 
# sub_features <- scale(sub_features)


breks <- seq(-1,1,length.out = 512)

histograms <- sapply(res, function(x){
  x1 <- x[[1]]
  distribution <-  tibble(x = x1[[1]], y = x1[[2]], z = x1[[3]])
  hist(distribution$y, breaks = breks, plot = FALSE)$counts/init_size
})

histograms <- t(histograms)







# model heatmaps
# model <- keras_model_sequential() 
# model %>%
#   layer_conv_2d(input_shape = c(100, 100, 1), filters = 12, kernel_size = 11, strides = 3) %>%
#   layer_batch_normalization() %>%
#   layer_activation_relu() %>%
#   layer_conv_2d(filters = 5, kernel_size = 5, strides = 2) %>%
#   layer_batch_normalization() %>%
#   layer_activation_relu() %>%
#   layer_flatten() %>%
#   layer_dense(units = 200, activation = "relu") %>%
#   layer_batch_normalization() %>%
#   layer_dense(units = 100, activation = "relu") %>%
#   layer_batch_normalization() %>%
#   layer_dense(units = 1, activation = "linear")


model <- keras_model_sequential()
model %>%
  layer_gaussian_noise(stddev = 1e-5) %>%
  layer_conv_1d(input_shape = c(511, 1), filters = 4, kernel_size = 7, strides = 3,
                kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_activation_leaky_relu() %>%
  layer_conv_1d(filters = 8, kernel_size = 7, strides = 2,
                kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_conv_1d(filters = 8, kernel_size = 5, strides = 2,
                kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_flatten() %>%
  layer_dense(units = 200, activation = "relu",
              kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_batch_normalization() %>%
  layer_alpha_dropout(rate = 0.2) %>%
  layer_dense(units = 100, activation = "relu",
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_alpha_dropout(rate = 0.2) %>%
  layer_dense(units = 50, activation = "relu",
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dense(units = 1, activation = "linear")



model %>% compile(
  loss = "mse",
  optimizer = optimizer_adam(lr = 1e-4), metrics = c("mae"))



indexes = createDataPartition(params_df$surv, p = .9, list = F)
indexes = sample(length(params_df$surv))[indexes]


targets_ori <- sapply(res, function(x){
  s <- x[[2]][1]
  e <- x[[2]][2]
  c(surv = s/init_size, eras = log2((init_size - s)/e ) )
})
targets_ori <- t(targets_ori)

means <- apply(targets_ori, 2, mean)
targets <- t(t(targets_ori) - means)
sds <- apply(targets, 2, sd)
targets <- t(t(targets) / sds)
targets <- targets[,1]




# predictors <- sub_features

xtrain = histograms[indexes,]

indexes_val = createDataPartition(1:length(indexes), p = .7, list = F)
indexes_val = sample(length(indexes))[indexes_val]

xval = xtrain[-indexes_val,]
xval <- array_reshape(unlist(xval), dim = c(dim(xval), 1))
xtrain = xtrain[indexes_val,]
xtrain <- array_reshape(unlist(xtrain), dim = c(dim(xtrain), 1))


xtest = histograms[-indexes,]
xtest <- array_reshape(unlist(xtest), dim = c(dim(xtest), 1))


# targets <- targets
ytrain = targets[indexes]
yval = ytrain[-indexes_val]
ytrain = ytrain[indexes_val]


ytest = targets[-indexes]
history <- model %>% fit(x = xtrain, y = ytrain, epochs = 100, verbose = 1, validation_data = list(xval, yval),
                         callbacks = list(
                           callback_early_stopping(patience = 20, restore_best_weights = T),
                           callback_reduce_lr_on_plateau(factor = 0.05))
)



scores = model %>% evaluate(xtest, ytest, verbose = 0)
print(scores)





whole_ds_pred <- model %>% predict(array_reshape(histograms,
                                                 dim = c(dim(histograms), 1)))


# colnames(targets) <- c("surv", "eras")
# 
# 
# trans_w_pred <- as.data.frame(t((t(whole_ds_pred) * sds) + means))
# colnames(trans_w_pred) <- c("surv_p", "eras_p")

trans_w_pred <- tibble(surv_p = whole_ds_pred*sds[1] + means[1])
# trans_w_pred <- tibble(surv_p = whole_ds_pred)


plot_d <- bind_cols(as.data.frame(targets_ori), trans_w_pred, group = 1:dim(whole_ds_pred)[1] %in% indexes )

plot_d %>%
  ggplot(aes(surv, surv_p)) + geom_point(  aes(col = group) ,alpha = 0.1 ) + geom_smooth(aes(group = as.factor(group))) +
  geom_abline(slope = 1, col = "red", size = 2)

# plot_d %>%
#   ggplot(aes(eras, eras_p)) + geom_point(  aes(col = group), alpha=0.3  ) + geom_smooth(aes(group = as.factor(group))) +
#   geom_abline(slope = 1, col = "red", size = 2)

# plot_d %>% ggplot(aes((surv - surv_p)/mean(surv), (eras - eras_p)/mean(eras))) + geom_point(  aes(col = group)  )

plot_d %>% ggplot(aes(  (surv - surv_p)/mean(surv)  )) + geom_histogram(  aes(fill = group), position = "identity", alpha = 0.5)
# plot_d %>% ggplot(aes(   (eras - eras_p)/mean(eras)   )) + geom_histogram(  aes(fill = group), position = "identity", alpha = 0.5)




ori_p <- sapply(list(ass_ori), function(x){
  x <- x
  x[x < -1] <- -1
  cut_val <- hist(x, breaks = breaks, plot = FALSE)
  cut_val$counts/init_size
})

preds <- model %>% predict(array_reshape(ori_p, c(1, dim(ori_p))))

trans_preds <- (preds * means) + sds

(1 - trans_preds[1])/(2^trans_preds[2] - 1)







# 
# 
# comb_frame %>% ggplot(aes(surv, sd_2)) + geom_point(aes(colour = pseudo_eras), alpha = 0.15, size = 2, shape = 16) +
#   geom_quantile() + scale_color_viridis(option = "magma")
# 
# 
# 
# covs <- cov(comb_frame)
# covs[covs > 0.8] <- 0
# cors <- cor(comb_frame, method = "p")
# 
# heatmap(x = (abs(cors)), scale = "none", Colv = NA, Rowv = NA)
# 
# 
# 
# cors_covs <- cov(bind_cols(comb_frame$surv, wh_cors))
# 
# heatmap(x = log(abs(cors_covs)), scale = "none", Colv = NA, Rowv = NA)
# 

# #super cool plot
# ggplot(distr_pca, aes(x = .panel_x, y = .panel_y) ) + 
#   geom_autodensity(position = "identity", alpha = 0.15, size = 1, aes(group = as_factor(iter), col = as_factor(iter)) ) +
#   geom_density2d_filled(contour_var = "ndensity" , adjust = 0.5) +
#   geom_point(alpha = 0.4, shape = 16, size = 1.5, aes(col = as_factor(iter))) + 
#   facet_matrix(vars(-iter), layer.diag = 1, layer.upper = 2)
# 

