
print(paste("Starting LIN4 |", Sys.time()))

# Recipe: Data Preprocessing
lin_rec <- recipe(BMI ~ .,
                  data = dt_train) %>%
  step_log(BMI, skip = TRUE) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_numeric_predictors())

# Model Specification
lin_spec <- linear_reg(penalty = tune(),
                       mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("regression")

# Workflow
lin_wflow <- workflow() %>% 
  add_recipe(
    lin_rec,
    blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = TRUE)
  ) %>% 
  add_model(lin_spec)


# Hyperparameter Tuning
start_time = Sys.time()

unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

cl <- makePSOCKcluster(6)
registerDoParallel(cl)

lin_tune <- tune_grid(object = lin_wflow,
                      resamples = folds,
                      grid = grid_latin_hypercube(mixture(), penalty(),
                                                  size = 100),
                      metrics = eval_metrics)

stopCluster(cl) 
unregister_dopar()

end_time = Sys.time()
print(end_time - start_time)

saveRDS(object = lin_tune, 
        file = paste0("LIN4_tune.rds"))

# Fitting the final model
lin_final_fit <- lin_wflow %>%
  finalize_workflow(select_best(lin_tune, metric = "rsq")) %>%
  last_fit(dt_split)

# Get R^2
lin_final_fit %>% 
  collect_predictions() %>% 
  rsq(truth = BMI, estimate = .pred) %>% 
  print()

saveRDS(object = lin_final_fit, 
        file = paste0("LIN4_fit.rds"))

print(paste("LIN4 done |", Sys.time()))