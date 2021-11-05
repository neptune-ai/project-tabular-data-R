library(neptune)
library(dplyr)
library(caret)
library(RCurl)
library(xgboost)
library(ggplot2)
# install.packages("neptune", dependencies = TRUE)

# (neptune) connect to the project
neptune::init_neptune(api_token    = "ANONYMOUS",
                      project_name = "common/project-tabular-data-R"
                      )

model_params <- list(
  eta =  0.3,
  gamma =  0.0001,
  max_depth =  2,
  colsample_bytree =  0.85,
  subsample =  0.9,
  objective = "reg:squarederror",
  eval_metric = c("rmse")
)

# (neptune) create new run
neptune::create_experiment(name   = "R-XGB-example",
                           params = model_params,
                           tags   = c("R"),
                           upload_source_files = c("neptune-xgb.R")
                           )

# (neptune) append tag to the run
neptune::append_tag(tag = "r-training")

data <- read.csv("./data/train.csv")

check_any_not_na <- function(x) {
  !all(is.na(x))
}

data <-
  data %>%  select_if(is.numeric) %>%  select_if(check_any_not_na)
y <- data$SalePrice
X <- data %>% select(-one_of("SalePrice"))
index <- createDataPartition(y, p = .85,
                             list = FALSE,
                             times = 1)

tmp_y <- y[index]
test_y <- y[-index]
tmp_X <- X[index,]
test_X <- X[-index,]

index <- createDataPartition(tmp_y,
                             p = .85,
                             list = FALSE,
                             times = 1)

train_y <- tmp_y[index]
valid_y <- tmp_y[-index]
train_X <- tmp_X[index,]
valid_X <- tmp_X[-index,]

pp <- preProcess(train_X, method = "medianImpute")
train_X <- predict(pp, train_X)
valid_X <- predict(pp, valid_X)
test_X <- predict(pp, test_X)

# (neptune) log dataset sizes
neptune::set_property(property = "data/train_size", value = nrow(train_X))
neptune::set_property(property = "data/valid_size", value = nrow(valid_X))
neptune::set_property(property = "data/test_size", value = nrow(test_X))

xgb_train = xgb.DMatrix(data = as.matrix(train_X), label = train_y)
xgb_test = xgb.DMatrix(data = as.matrix(test_X))
xgb_valid = xgb.DMatrix(data = as.matrix(valid_X), label = valid_y)

model = xgb.train(
  data = xgb_train,
  nrounds = 50,
  watchlist = list(train = xgb_train,
                   valid = xgb_valid),
  params = model_params
)

# (neptune) log training metrics
for (i in 1:nrow(model$evaluation_log)) {
  neptune::log_metric("training/train_rmse", model$evaluation_log$train_rmse[i])
  neptune::log_metric("training/valid_rmse", model$evaluation_log$valid_rmse[i])
}

# (neptune) log model
save(model, file = "model.Rdata")
neptune::log_artifact("model.Rdata")

# (neptune) log train and test scores
neptune::set_property(property = "score/train_rmse", value = RMSE(train_y, predict(model, xgb_train)))
neptune::set_property(property = "score/test_rmse", value = RMSE(test_y, predict(model, xgb_test)))


fi <-
  xgb.importance(feature_names = model$feature_names, model = model)

p <- ggplot(fi) + geom_text(aes(label = Feature,
                                y = Gain,
                                x = Frequency))
ggsave("FI_freq_vs_gain.png", p)

# (neptune) log feature importance chart as image
neptune::log_image("feature importance", "FI_freq_vs_gain.png")

p <- ggplot(fi) + geom_bar(aes(x = Feature,
                               fill = Feature,
                               y = Gain),
                           stat = "identity") + coord_flip() + guides(fill="none")

ggsave("FI_freq_vs_gain.png", p)

# (neptune) log feature importance chart as image
neptune::log_image("feature importance barplot Gain", "FI_freq_vs_gain.png")

# (neptune) close the run
neptune::stop_experiment()