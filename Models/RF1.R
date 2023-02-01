
print(paste("Starting RF1 |", Sys.time()))

# Recipe: Data Preprocessing
rf_rec <- recipe(BMI ~ .,
                 data = dt_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_zv(all_predictors())

# Model Specification
rf_spec <- rand_forest(mtry = tune(),
                       trees = tune(),
                       min_n = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("regression")

# Workflow
rf_wflow <- workflow() %>% 
  add_recipe(
    rf_rec,
    blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = TRUE)
  ) %>% 
  add_model(rf_spec)

# Hyperparameter Tuning
start_time = Sys.time()

unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

cl <- makePSOCKcluster(6)
registerDoParallel(cl)

rf_tune <- tune_grid(object = rf_wflow,
                     resamples = folds,
                     grid = grid_latin_hypercube(finalize(mtry(), dt_train),
                                                 min_n(), trees(), 
                                                 size = 50),
                     control = control_grid(save_pred = TRUE,
                                            save_workflow = TRUE),
                     metrics = eval_metrics)

stopCluster(cl)
unregister_dopar()

end_time = Sys.time()
print(end_time - start_time)

saveRDS(object = lin_tune, 
        file = paste0("RF1_tune_", 
                      substr(Sys.time(), 1, 19) %>% 
                        str_replace_all(":", "-"),
                      ".rds"))

# Fitting the final model
rf_final_fit <- rf_wflow %>%
  finalize_workflow(select_best(rf_tune, metric = "rsq")) %>%
  last_fit(dt_split)

saveRDS(object = lin_final_fit, 
        file = paste0("RF1_fit_",
                      substr(Sys.time(), 1, 19) %>% 
                        str_replace_all(":", "-"),
                      ".rds"))

print(paste("RF1 done |", Sys.time()))
