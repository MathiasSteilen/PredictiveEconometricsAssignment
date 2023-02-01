
print(paste("Starting GB1 |", Sys.time()))

# Recipe: Data Preprocessing
gb_rec <- recipe(BMI ~ .,
                 data = dt_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% 
  step_zv(all_predictors())

# Model Specification
gb_spec <- boost_tree(trees = tune(),
                      tree_depth = tune(),
                      min_n = tune(),
                      loss_reduction = tune(),
                      sample_size = tune(),
                      mtry = tune(),
                      learn_rate = tune()) %>%
  set_engine("xgboost", importance = "impurity") %>%
  set_mode("regression")

# Workflow
gb_wflow <- workflow() %>% 
  add_recipe(
    gb_rec,
    blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = TRUE)
  ) %>% 
  add_model(gb_spec)

# Hyperparameter Tuning
set.seed(1)
start_time = Sys.time()

unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

cl <- makePSOCKcluster(6)
registerDoParallel(cl)

gb_tune <- tune_grid(object = gb_wflow,
                     resamples = folds,
                     grid = grid_latin_hypercube(trees(), tree_depth(), 
                                                 min_n(), loss_reduction(), 
                                                 sample_size = sample_prop(),
                                                 finalize(mtry(), dt_train),
                                                 learn_rate(), size = 200),
                     metrics = eval_metrics,
                     control = control_grid(save_pred = TRUE,
                                            save_workflow = TRUE))

stopCluster(cl)
unregister_dopar()

end_time = Sys.time()
print(end_time - start_time)

saveRDS(object = gb_tune, 
        file = paste0("GB1_tune.rds"))

# Fitting the final model
gb_final_fit <- gb_wflow %>%
  finalize_workflow(select_best(gb_tune, metric = "rsq")) %>%
  last_fit(dt_split)

# Get R^2
gb_final_fit %>% 
  collect_predictions() %>% 
  rsq(truth = BMI, estimate = .pred) %>% 
  print()

saveRDS(object = gb_final_fit, 
        file = paste0("GB1_fit.rds"))

print(paste("GB1 done |", Sys.time()))
