# modelr tutorial
#
# https://github.com/tidyverse/modelr
# 

library(modelr)

### 1. Partitioning and sampling
# resample()
head(mtcars)
rs <- resample(mtcars, 1:20)
as.data.frame(rs)

# resample_partition()
part <- resample_partition(mtcars, c(test = .3, train = .7))
part

# bootstrap()
boot <- bootstrap(mtcars, 5)
boot
boot$strap # all sample idx

# crossv_kfold()
cvk <- crossv_kfold(mtcars, 5)
cvk$train
cvk$test


### 2. Metrics

linear_model <- lm(mpg ~ wt, data = mtcars)

# rmse()
rmse(linear_model, mtcars)

# rsquare()
rsquare(linear_model, mtcars)

# mae()

# qae()


### 3. Interacting and accessing models

df <- tibble(
  x = sort(runif(100)),
  y = 5 * x + .5 * x^2 + 3 + rnorm(length(x))
)

linear_model <- lm(y ~ x, data = df)

# add_predictions() - append predicted y on df
df %>% add_predictions(linear_model)

# add_residuals() - append residuals on df
df %>% add_residuals(linear_model)

# data_grid() - similar to expand.grid()
