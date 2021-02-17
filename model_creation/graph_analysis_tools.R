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
library(data.table)

# graph kernel =e^(-(1 / (2 * sigma^2))*|x|^2) sigma=1


mix_mat <- function(atrr_list){
  colnames(atrr_list) <- c("c1", "c2", "V3")
  mmax <- max(atrr_list[,1:2])
  fillingframe <- cbind(c1 = 1:mmax, c2 = 1:mmax, V3 = rep(0, mmax))
  fillingframe <- tibble::as_tibble(fillingframe)
  atrr_list <- dplyr::bind_rows(atrr_list, fillingframe)
  sum_list <- atrr_list %>% dplyr::group_by(c1,c2) %>% dplyr::summarise(sum = sum(V3), .groups = "drop" )
  mymat <- tidyr::spread(sum_list, c1, sum)
  s <- as.matrix(mymat[,-1])
  s[is.na(s)] <- 0
  s <- s + t(s)
  s <- s/sum(s)
  s
}


read_dist_to_graphs <- function(){
  
  dir <- choose.dir()
  setwd(dir = dir)
  # dir("./distances")
  colors <- list()
  for (i in 1:3) {
    data = read_csv(paste0("./split/C", i,"/cells.csv"), col_names = FALSE)
    data = t(data)
    colnames(data) <- c("X", "Y")
    data <- as_tibble(data)
    data <- data %>% mutate(namez = paste(X,Y))
    colors[[i]] = data
  }
  
  positions <- list()
  distances = list()
  files = dir("./distances")
  for (i in 1:length(files)) {
    data = read_csv(paste0("./distances/", files[i]), col_names = FALSE)
    colnames(data) <- paste(data[1,],data[2,])
    positions[[i]] = data[1:2,]
    data <- data[3:dim(data)[1],]
    distances[[i]] = data
    
  }
  data = read_csv(paste0("./cell_positions.csv"), col_names = FALSE)
  data = t(data)
  colnames(data) <- c("X", "Y")
  data <- as_tibble(data)
  data <- data %>% dplyr::mutate(namez = paste(X, Y))
  data <- tibble::add_column(data, color = 0)
  
  for (i in 1:3) {
    data$color[data$namez %in% colors[[i]]$namez] = i
  }
  
  for (i in 1:length(files)) {
    pos = colnames(distances[[i]])
    
    col = data %>% dplyr::filter(namez %in% pos) %>% pull(color)
    
    colnames(distances[[i]]) = col
  }
  
  graphs <- list()
  for (i in 1:length(distances)) {
    dist_mat <- as.matrix(distances[[i]])
    ad_mat <- (2500 - dist_mat)/2500
    ad_mat[ad_mat < 0] <- 0
    colnames(ad_mat) <- 1:dim(ad_mat)[2]
    graphs[[i]] <- graph.adjacency(ad_mat, weighted = TRUE, diag = F)
    graphs[[i]] <- as.undirected(graphs[[i]])
    graphs[[i]] <- set.vertex.attribute(graphs[[i]], "X", value = positions[[i]] %>% slice(1) %>% unlist())
    graphs[[i]] <- set.vertex.attribute(graphs[[i]], "Y", value = positions[[i]] %>% slice(2) %>% unlist())
    graphs[[i]] <- set.vertex.attribute(graphs[[i]], "Color", value = colnames(distances[[i]]))
  }
  graphs
}


reduce_edges <- function(graph){
  # check intersections
  orientation <- function(p, q, r){
    val = ((q[2] - p[2]) * (r[1] - q[1])) - ((q[1] - p[1]) * (r[2] - q[2]))
    res <- val/abs(val)
    if (is.na(res)) {
      return(0) 
    }
    return(res)
  }
  
  onSegment <- function(p, q, r){ 
    if ((q[1] <= max(p[1], r[1])) & (q[1] >= min(p[1], r[1])) & (q[2] <= max(p[2], r[2])) & (q[2] >= min(p[2], r[2]))) { 
      return(T)}else{return(F)}
  }
  
  do_intersect <- function(p1,q1,p2,q2){
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1)
    
    if ((o1 != o2) & (o3 != o4)) {
      return(T)
    }
    
    
    # Special Cases 
    
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
      if ((o1 == 0) & onSegment(p1, p2, q1)) {
        return(T)
      }
    
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
      if ((o2 == 0) & onSegment(p1, q2, q1)) {
        return(T)
      }
    
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
      if ((o3 == 0) & onSegment(p2, p1, q2)) {
        return(T)
      }
    
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
      if ((o4 == 0) & onSegment(p2, q1, q2)) {
        return(T)
      }
    
    # If none of the cases 
    {return(F)}
  }
  intersect_edge_comp <- function(i, vertices, edges){
    p1 <- vertices[edges[i,1],]
    q1 <- vertices[edges[i,2],]
    elength <- dim(edges)[1]
    for (j in 1:elength) {
      if (sum(edges[j,1:2] %in% edges[i,1:2]) < 1) {
        p2 <- vertices[edges[j,1],]
        q2 <- vertices[edges[j,2],]
        val <-  do_intersect(p1,q1,p2,q2)
        if (val) {
          if (edges[i,3] < edges[j,3]) {
            return(1)
          }
        }
      }
    }
    return(0)
  }
  
  
  edges <- ends(graph, es = 1:length(E(graph)))
  edges <- as.data.frame(edges)
  edges[,3] <- get.edge.attribute(graph,name = "weight")
  vertices <- base::cbind(get.vertex.attribute(graph)$X,
                    get.vertex.attribute(graph)$Y)
  rownames(vertices) <- get.vertex.attribute(graph, "name")
  library(foreach)
  library(doParallel)
  cores = detectCores()
  cl <- makeCluster(cores[1] - 1) #not to overload your computer
  registerDoParallel(cl)
  
  finalMatrix <- foreach(i = 1:dim(edges)[1], .combine = cbind, .verbose = F) %dopar% {
    tempMatrix = intersect_edge_comp(i, vertices, edges)
  }
  return(finalMatrix)
}


reduce_specific_edges <- function(graph, s_edges){
  #edges, as vector,of boolean
  
  # check intersections
  orientation <- function(p, q, r){
    val = ((q[2] - p[2]) * (r[1] - q[1])) - ((q[1] - p[1]) * (r[2] - q[2]))
    res <- val/abs(val)
    if (is.na(res)) {
      return(0)
    }
    return(res)
  }
  
  
  do_intersect <- function(p1,q1,p2,q2){
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1)
    
    if ((o1 != o2) & (o3 != o4)) {
      return(T)
    }
    
    
    # Special Cases 
    
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
    #  if ((o1 == 0) & onSegment(p1, p2, q1)) {
    #    return(T)
    #  }
    
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
    #  if ((o2 == 0) & onSegment(p1, q2, q1)) {
    #    return(T)
    #  }
    
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
    #  if ((o3 == 0) & onSegment(p2, p1, q2)) {
    #    return(T)
    #  }
    
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
    #  if ((o4 == 0) & onSegment(p2, q1, q2)) {
    #    return(T)
    #  }
    
    # If none of the cases 
    return(F)
  }
  
  
  intersect_edge_comp <- function(i, vertices, edges){
    p1 <- vertices[edges[i,1],]
    q1 <- vertices[edges[i,2],]
    elength <- dim(edges)[1]
    for (j in 1:elength) {
      if (sum(edges[j,1:2] %in% edges[i,1:2]) < 1) {
        p2 <- vertices[edges[j,1],]
        q2 <- vertices[edges[j,2],]
        val <-  do_intersect(p1,q1,p2,q2)
        if (val) {
          if (edges[i,3] < edges[j,3]) {
            return(1)
          }
        }
      }
    }
    return(0)
  }
  
  
  edges <- ends(graph, es = 1:length(E(graph)))
  edges <- as.data.frame(edges)
  edges[,3] <- get.edge.attribute(graph,name = "weight")
  vertices <- cbind(get.vertex.attribute(graph)$X,
                    get.vertex.attribute(graph)$Y)
  rownames(vertices) <- get.vertex.attribute(graph, "name")
  library(foreach)
  library(doParallel)
  cores = detectCores()
  cl <- makeCluster(cores[1] - 1) #not to overload your computer
  registerDoParallel(cl)
  
  finalMatrix <- foreach(i = which(s_edges), .combine = cbind, .verbose = F) %dopar% {
    tempMatrix = intersect_edge_comp(i, vertices, edges)
  }
  return(finalMatrix)
}


reduce_edges_splitting <- function(graph, verbose=T){
  commu <- cluster_fast_greedy(graph)
  membership <- commu$membership
  if (verbose) {
    print(date())
    c_lens <- c()
    for (i in 1:length(commu)) c_lens[i] <- length(commu[[i]])
    pb <- txtProgressBar(style = 3, max = sum(c_lens))
    setTxtProgressBar(pb, 0)
  }
  
  edgelist <- get.edgelist(graph)
  edgelist <- apply(edgelist,1, function(x)paste(x[1], x[2], sep = "e"))
  reductlist <- rep(0, length(edgelist))
  for (s in 1:length(commu)) {
    subgraph <- induced_subgraph(graph, commu[[s]])
    if (length(E(subgraph)) > 1) {
      if (length(E(subgraph)) > 1000) {
        rval <- reduce_edges_splitting(subgraph, verbose = F)
      }else{
        rval <- reduce_edges(subgraph)
      }
      sub_edgelist <- apply(get.edgelist(subgraph),1, function(x)paste(x[1], x[2], sep = "e"))
      
      reductlist[edgelist %in% sub_edgelist] <- rval
    }
    
    if (verbose) setTxtProgressBar(pb, sum(c_lens[1:s]))
  }
  graph <- delete.edges(graph, edges = which(reductlist > 0))
  crss <- reduce_specific_edges(graph,s_edges =  crossing(commu, graph))
  reduced_edgelist <- apply(get.edgelist(graph),1, function(x)paste(x[1], x[2], sep = "e"))
  reductlist[edgelist %in% reduced_edgelist[crossing(commu, graph)]] <- crss
  return(reductlist)
}


assortativity_weighted <- function(graph, val){
  val <- as.integer(val)
  c1 <- val[igraph::get.edgelist(graph, names = F)[,1]]
  c2 <- val[igraph::get.edgelist(graph, names = F)[,2]]
  
  atrr_list <- cbind(c1,c2, igraph::get.edge.attribute(graph, name = "weight"))
  atrr_list <- as_tibble(atrr_list)
  s <- mix_mat(atrr_list)
  if (dim(s)[1] == 1 | sum(s %*% s) == 1) return(1)
  (sum(diag(s)) - sum(s %*% s)) / (1 - sum(s %*% s))
}


simulate_expansion <- function(graph, fraction=0.1){
  nlen <- length(V(graph))
  rand_nodes <- sample(1:nlen, ceiling(nlen*fraction))
  oris <- rep(100000, nlen)
  oris[rand_nodes] <- rand_nodes
  oris <- as.factor(oris)
  wframe <- igraph::distances(graph, v = rand_nodes, weights = (2 - get.edge.attribute(graph, "weight")))
  expanded <- apply(wframe,2, function(x){
    order(x)[1]
  })
  list(origins = oris, expanded = expanded)
}


simulate_expansion_of_fraction <- function(graph, survivor_fraction=0.3, expanding_fraction=0.1){
  nlen <- length(V(graph))
  surv_nodes <- sample(1:nlen, ceiling(nlen*survivor_fraction))
  survivors <- rep(100000, nlen)
  survivors[surv_nodes] <- surv_nodes
  ori_nodes <- sample(surv_nodes, ceiling(length(surv_nodes)*expanding_fraction))
  oris <- rep(100000, nlen)
  oris[ori_nodes] <- ori_nodes
  oris <- as.factor(oris)
  wframe <- igraph::distances(graph, v = ori_nodes, weights = (2 - get.edge.attribute(graph, "weight")))
  expanded <- apply(wframe,2, function(x){
    order(x)[1]
  })
  final <- expanded
  non_exp <- surv_nodes[!surv_nodes %in% ori_nodes]
  final[non_exp] <- max(expanded) + (1:length(non_exp))
  list(survivors = survivors, origins = oris, expanded = expanded, final = final)
}



simulate_expansion_of_fraction_2 <- function(graph, survivor_fraction=0.3, expanding_fraction=0.1){
  nlen <- length(V(graph))
  surv_nodes <- sample(1:nlen, ceiling(nlen*survivor_fraction))
  survivors <- rep(NA, nlen)
  survivors[surv_nodes] <- surv_nodes
  ori_nodes <- sample(surv_nodes, ceiling(length(surv_nodes)*expanding_fraction))
  oris <- rep(NA, nlen)
  oris[ori_nodes] <- ori_nodes
  oris <- as.factor(oris)
  cl = makeCluster(7)
  on.exit(stopCluster(cl))
  clusterExport(cl, c("nlen", "graph"), envir = environment())
  
  
  non_exp <- surv_nodes[!surv_nodes %in% ori_nodes]
  E(graph)[incident(graph, non_exp)]$weight <- E(graph)[incident(graph, non_exp)]$weight * 0.5
  
  
  conn = parSapply(cl, X = ori_nodes, FUN = function(x){
    pers <- rep(0, nlen)
    pers[x] <- 1
    igraph::page_rank(graph, personalized = pers, damping =  0.5, weights = NULL)$vector
  })
  
  
  expanded <- parApply(cl, conn,1, function(x){
    order(x, decreasing = T)[1]
  })
  
  
  dead <- colMeans(distances(graph, v = sort(ori_nodes)) == Inf) == 1
  expanded[dead] <- NA
  final <- expanded
  final[non_exp] <- max(expanded, na.rm = T) + (1:length(non_exp))
  list(survivors = survivors, origins = oris, expanded = expanded, final = final)
}




simulate_expansion_of_fraction_cheap <- function(graph, survivor_fraction=0.3, expanding_fraction=0.1, cols){
  nlen <- length(V(graph))
  surv_nodes <- sample(1:nlen, ceiling(nlen*survivor_fraction))
  survivors <- rep(NA, nlen)
  survivors[surv_nodes] <- surv_nodes
  ori_nodes <- sample(surv_nodes, ceiling(length(surv_nodes)*expanding_fraction))
  oris <- rep(NA, nlen)
  oris[ori_nodes] <- ori_nodes
  oris <- cols[oris]
  
  non_exp <- surv_nodes[!surv_nodes %in% ori_nodes]
  E(graph)[incident(graph, non_exp)]$weight <- E(graph)[incident(graph, non_exp)]$weight * 0.5
  
  
  conn = sapply(X = sort(na.exclude(unique(oris))), FUN = function(x){
    pers <- rep(0, nlen)
    pers[oris == x] <- 1
    igraph::page_rank(graph, personalized = pers, damping =  0.5, weights = NULL)$vector
  })
  
  
  expanded <- apply(conn, 1, function(x){
    order(x, decreasing = T)[1]
  })
  
  
  dead <- colMeans(distances(graph, v = ori_nodes) == Inf) == 1
  expanded[dead] <- NA
  final <- expanded
  final[non_exp] <- cols[max(expanded, na.rm = T) + (1:length(non_exp))]
  list(survivors = survivors, origins = oris, expanded = expanded, final = final)
}
  




assortativity_local_par <- function(graph, val, alpha = 0.2){
  cl = makeCluster(7)
  on.exit(stopCluster(cl))
  graph_l <- length(V(graph))
  proceed <- graph_l == length(val)
  if (!proceed) stop("length of data differs")
  val <- as.integer(val)
  e2 <- igraph::get.edgelist(graph, names = F)
  e1 <- e2[,1]
  e2 <- e2[,2]
  atrr_list <- cbind(val[e1], val[e2], rep(1, length(val[e1])))
  colnames(atrr_list) <- c("c1", "c2", "V3")
  atrr_list <- dplyr::as_tibble(atrr_list)
  s <- mix_mat(atrr_list)
  agg <- sum(s %*% s)
  if (agg == 1) agg <- 0.9999999999
  #sameness bool
  c_bool <- val[c(e1,e2)] == val[c(e2,e1)]
  deg <- as.numeric(igraph::degree(graph)[c(e1,e2)])
  assorts <- rep(0, graph_l)
  cc <- c(e1,e2)
  clusterExport(cl, c("graph_l", "deg", "graph", "alpha", "cc"), envir = environment())
  assorts = parSapply(cl, X = 1:graph_l, FUN = function(x){
    pers <- rep(0, graph_l)
    pers[x] <- 1
    Vpage <- igraph::page_rank(graph, personalized = pers, damping =  alpha, weights = NULL)$vector
    Vpage <- Vpage[cc]/deg
    Vpage[Vpage < 0] <- 0
    sum(Vpage[c_bool])/sum(Vpage)
  })
  
  assorts <- (assorts - agg)/(1 - agg)
  assorts
}
assortativity_local_par <- compiler::cmpfun(assortativity_local_par)





fast_assorts_p_1 <- function(graph, alpha = 0.4){
  cl = makeCluster(7)
  on.exit(stopCluster(cl))
  graph_l <- length(V(graph))
  e2 <- igraph::get.edgelist(graph, names = F)
  e1 <- e2[,1]
  e2 <- e2[,2]
  cc <- c(e1, e2)
  deg <- as.numeric(igraph::degree(graph)[c(e1,e2)])
  clusterExport(cl, c("graph_l", "deg", "graph", "alpha", "cc"), envir = environment())
  assorts_ori = parSapply(cl, X = 1:graph_l, FUN = function(x){
    pers <- rep(0, graph_l)
    pers[x] <- 1
    Vpage <- igraph::page_rank(graph, personalized = pers, damping =  alpha, weights = NULL)$vector
    Vpage <- Vpage[cc]/deg
    Vpage
  })
  assorts_ori[assorts_ori < 0] <- 0
  return(list(assorts_ori, e1, e2))
}

fast_assorts_p_1 <- compiler::cmpfun(fast_assorts_p_1)



fast_assorts_p_2 <- function(val, assorts, e1, e2){
  val <- as.integer(val)
  atrr_list <- cbind(val[e1], val[e2], rep(1, length(val[e1])))
  colnames(atrr_list) <- c("c1", "c2", "V3")
  atrr_list <- dplyr::as_tibble(atrr_list)
  s <- mix_mat(atrr_list)
  agg <- sum(s %*% s)
  if (agg == 1) agg <- 0.9999999999
  #sameness bool
  c_bool <- val[c(e1,e2)] == val[c(e2,e1)]
  c_bool <- c_bool*1
  assorts <- colSums(assorts*c_bool)/colSums(assorts)
  assorts <- (assorts - agg)/(1 - agg)
  assorts
}

fast_assorts_p_2 <- compiler::cmpfun(fast_assorts_p_2)







two_step_xgb_tune <- function(recipe, data_train){
  xgb_spec_1 <- boost_tree(
    trees = 1000, 
    tree_depth = 10, min_n = 10, 
    loss_reduction = tune(),                     ## first three: model complexity
    sample_size = 0.5, mtry = 10,         ## randomness
    learn_rate = tune(),                         ## step size
  ) %>% 
    set_engine("xgboost") %>% 
    set_mode("regression")
  
  
  

  
  
  xgb_grid <- grid_latin_hypercube(
    loss_reduction(),
    learn_rate(),
    size = 50
  )
  
  tune_wf <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(xgb_spec_1)
  
  data_folds <- bootstraps(data_train, times = 10)
  
  
  doParallel::registerDoParallel(cores = parallel::detectCores() - 1)
  set.seed(234)
  xgb_res <- tune_grid(
    tune_wf,
    resamples = data_folds,
    grid = xgb_grid,
    control = control_grid(save_pred = TRUE)
  )
  print("Step one- done")
  collect_metrics(xgb_res)
  
  plt_1 <- xgb_res %>%
    collect_metrics() %>%
    filter(.metric == "rmse") %>%
    select(mean, learn_rate:loss_reduction ) %>%
    pivot_longer(learn_rate:loss_reduction,
                 values_to = "value",
                 names_to = "parameter"
    ) %>%
    ggplot(aes(value, mean, color = parameter)) +
    geom_point(alpha = 0.8, show.legend = FALSE) +
    facet_wrap(~parameter, scales = "free_x") +
    labs(x = NULL, y = "rmse")
  
  
  
  
  best <- as.numeric(select_best(xgb_res))
  

  
  xgb_spec_2 <- boost_tree(
    trees = 1000, 
    tree_depth = tune(), min_n = tune(), 
    loss_reduction = !!best[2],                     ## first three: model complexity
    sample_size = tune(), mtry = tune(),         ## randomness
    learn_rate = !!best[1],                         ## step size
  ) %>% 
    set_engine("xgboost") %>% 
    set_mode("regression")
  
  
  xgb_grid <- grid_latin_hypercube(
    tree_depth(),
    min_n(),
    sample_size = sample_prop(),
    finalize(mtry(), data_train),
    size = 200
  )
  
  tune_wf <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(xgb_spec_2)
  
  
  
  doParallel::registerDoParallel(cores = parallel::detectCores() - 1)
  set.seed(345)
  xgb_res <- tune_grid(
    tune_wf,
    resamples = data_folds,
    grid = xgb_grid,
    control = control_grid(save_pred = TRUE)
  )
  print("Step two- done")
  collect_metrics(xgb_res)
  
  plt_2 <- xgb_res %>%
    collect_metrics() %>%
    filter(.metric == "rmse") %>%
    select(mean, mtry:sample_size) %>%
    pivot_longer(mtry:sample_size,
                 values_to = "value",
                 names_to = "parameter"
    ) %>%
    ggplot(aes(value, mean, color = parameter)) +
    geom_point(alpha = 0.8, show.legend = FALSE) +
    facet_wrap(~parameter, scales = "free_x") +
    labs(x = NULL, y = "rmse")
  
  best_rmse <- select_best(xgb_res, "rmse")
  
  final_xgb <- finalize_workflow(
    tune_wf,
    best_rmse
  )
  list(final_xgb = final_xgb, plt_1 = plt_1, plt_2 = plt_2)
}
