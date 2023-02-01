
print(paste("Starting SVM5 |", Sys.time()))

# Recipe: Data Preprocessing
svm_rec <- recipe(BMI ~ .,
                    data = dt_train) %>%
  step_discretize(Age, Height, num_breaks = 5, min_unique = 3) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = tune()) %>% 
  step_zv(all_predictors())

# Model Specification
svm_spec <- svm_rbf(cost = tune(),
                           rbf_sigma = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

# Workflow
svm_wflow <- workflow() %>% 
  add_recipe(
    svm_rec,
    blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = TRUE)
  ) %>% 
  add_model(svm_spec)

# Hyperparameter Tuning
set.seed(1)
start_time = Sys.time()

unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

cl <- makePSOCKcluster(6)
registerDoParallel(cl)

svm_tune <- tune_grid(object = svm_wflow,
                             resamples = folds,
                             grid = grid_latin_hypercube(cost(), rbf_sigma(),
                                                         threshold(),
                                                         size = 200),
                             metrics = eval_metrics)

stopCluster(cl)
unregister_dopar()

end_time = Sys.time()
print(end_time - start_time)

saveRDS(object = svm_tune, 
        file = paste0("SVM5_tune.rds"))

# Fitting the final model
svm_final_fit <- svm_wflow %>%
  finalize_workflow(select_best(svm_tune, metric = "rsq")) %>%
  last_fit(dt_split)

# Get R^2
svm_final_fit %>% 
  collect_predictions() %>% 
  rsq(truth = BMI, estimate = .pred) %>% 
  print()

saveRDS(object = svm_final_fit, 
        file = paste0("SVM5_fit.rds"))

print(paste("SVM5 done |", Sys.time()))
