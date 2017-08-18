## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)

## ----preparation---------------------------------------------------------
setwd('C:/Users/Bangda/Desktop/kaggle/housing-price0806')
# setwd('C:/Users/bs2996/Downloads/housing-price0805')
library(tidyverse)
library(magrittr)
library(stringr)
library(e1071)
library(VIM)
library(mice)
library(gridExtra)
train <- read.csv('train.csv', header = TRUE, stringsAsFactors = FALSE)
test  <- read.csv('test.csv', header = TRUE, stringsAsFactors = FALSE)
dim(train)
dim(test)

## ----check data types----------------------------------------------------
# get the data type of each column
train_var_class <- sapply(train, class)
test_var_class  <- sapply(test, class) 
table(train_var_class)
table(test_var_class)
# numeric variables
train_var_class[train_var_class %in% c('integer', 'numeric')] %>% names()
# categorical variables
train_var_class[train_var_class %in% c('factor', 'character')] %>% names()

## ----view missing data---------------------------------------------------
train_na_stat <- apply(train, 2, function(.data) sum(is.na(.data)))
train_na_stat[train_na_stat > 0]
test_na_stat  <- apply(test, 2, function(.data) sum(is.na(.data)))
test_na_stat[test_na_stat > 0]
# get the variables contain NA
train_na_variable <- names(train_na_stat[train_na_stat > 0])
test_na_variable  <- names(test_na_stat[test_na_stat > 0])

## ----remove variables and update-----------------------------------------
train %<>%
  select(-Alley, -PoolQC, -Fence, -MiscFeature)
test %<>%
  select(-Alley, -PoolQC, -Fence, -MiscFeature)
# update
train_na_stat <- apply(train, 2, function(.data) sum(is.na(.data)))
test_na_stat  <- apply(test, 2, function(.data) sum(is.na(.data)))
train_na_variable <- names(train_na_stat[train_na_stat > 0])
test_na_variable  <- names(test_na_stat[test_na_stat > 0])

## ----visualize missing data----------------------------------------------
train[, colnames(train) %in% names(train_na_stat[train_na_stat > 0])] %>%
  aggr(prop = FALSE, combined = TRUE, sortVars = TRUE, cex.axis = .7)
test[, colnames(test) %in% names(test_na_stat[test_na_stat > 0])] %>%
  aggr(prop = FALSE, combined = TRUE, sortVars = TRUE, cex.axis = .7)

## ----get number of missing data by variables-----------------------------
train %>%
  apply(1, function(.row) sum(is.na(.row))) %>% 
  sort(x = ., decreasing = TRUE) %>%
  '[' (1:10)
test %>%
  apply(1, function(.row) sum(is.na(.row))) %>% 
  sort(x = ., decreasing = TRUE) %>%
  '[' (1:10)

## ----calculate mode of data----------------------------------------------
getMode <- function(x, na.rm = TRUE) {
  # get mode for character vector
  if (na.rm) {
    sort(table(x[!is.na(x)]), decreasing = TRUE)[1] %>% names()  
  } else {
    sort(table(x), decreasing = TRUE)[1] %>% names() 
  }
}

## ----get the data with na value------------------------------------------
train_na_set <- train[, colnames(train) %in% train_na_variable]
test_na_set  <- test[, colnames(test) %in% test_na_variable]

## ----function to fill na values------------------------------------------
fillNA <- function(x) {
  
  if (sum(is.na(x)) == 0) return(x)
  
  if (class(x) %in% c('integer', 'numeric')) {
    x[which(is.na(x))] <- median(x, na.rm = TRUE) 
  } else {
    x[which(is.na(x))] <- getMode(x, na.rm = TRUE)
  }
  x
}

## ----test the function---------------------------------------------------
table(train_na_set$MasVnrType)
train_na_set$MasVnrType %<>% fillNA()
table(train_na_set$MasVnrType)

## ----apply na fill method on data sets-----------------------------------
train_na_filled <- train
test_na_filled <- test
for (i in 1:ncol(train)) {
  train_na_filled[, i] <- fillNA(train[, i])
}
for (i in 1:ncol(test)) {
  test_na_filled[, i] <- fillNA(test[, i])
}

## ----visualize price distribution----------------------------------------
ggplot(train, aes(x = SalePrice)) + 
  geom_histogram(aes(y = ..density..), binwidth = 10000,
    color = 'black', fill = 'skyblue') +
  geom_density(aes(y = ..density..), size = 1, col = 'red')
skewness(train$SalePrice)

## ----visualize price distribution cond-----------------------------------
train %>%
  filter(SalePrice <= quantile(SalePrice, .97)) %>%
  ggplot(aes(x = SalePrice)) + 
  geom_histogram(aes(y = ..density..), binwidth = 5000,
    color = 'black', fill = 'skyblue') +
  geom_density(aes(y = ..density..), size = 1, col = 'red')
train %>%
  filter(SalePrice <= quantile(SalePrice, .97)) %$%
  skewness(SalePrice)

## ----visualize log-price distribution------------------------------------
train %>%
  mutate(logPrice = log(SalePrice)) %>%
  ggplot(aes(x = logPrice)) + 
  geom_histogram(aes(y = ..density..), binwidth = 0.05,
                 color = 'black', fill = 'skyblue') +
  geom_density(aes(y = ..density..), size = 1, col = 'red')

## ----calculate skewness of price-----------------------------------------
train %<>% mutate(logSalePrice = log(SalePrice))
train_na_filled %<>% mutate(logSalePrice = log(SalePrice))
skewness(train$logSalePrice)

## ----cluster variables---------------------------------------------------
# get the first 4 characters of names
prefix <- names(train) %>% str_sub(1, 3) %>% unique()
prefix_ref <- data.frame(prefix, stringsAsFactors = FALSE)
# count the variables with same prefix
prefix_ref$count <- sapply(prefix_ref$prefix, 
  function(pattern) str_detect(names(train), pattern) %>% sum())
# filter the count with at least 3 
prefix_ref %<>%
  filter(count >= 3)
# get the variables that could have groups
grouped_var_idx <- sapply(
  str_extract_all(paste0(names(train)), 
    paste(prefix_ref$prefix, collapse = '|')), 
  function(num) length(num) > 0)
names(train)[grouped_var_idx]

## ----visualize logSalePrice to LotArea - OverallQual---------------------
ggplot(train_na_filled) + 
  geom_point(aes(x = log(LotArea), y = logSalePrice, 
    col = factor(OverallQual)), alpha = I(1/4))

## ----visualize logSalePrice to logTotalBsmtSF - BsmtCond-----------------
train_na_filled %>%
  mutate(BsmtCond = factor(BsmtCond, levels = c('TA', 'Fa', 'Gd', 'Po'))) %>%
  ggplot() + 
  geom_point(aes(x = log(TotalBsmtSF), y = logSalePrice, col = BsmtCond,
    size = BsmtCond), alpha = I(1/4))

## ----visualize kitchen related variables---------------------------------
train_na_filled %>%
  mutate(KitchenQual = factor(KitchenQual, levels = c('TA', 'Fa', 'Gd', 'Ex'))) %>%
  ggplot(aes(x = factor(KitchenAbvGr), y = logSalePrice)) + 
  geom_boxplot(aes(col = KitchenQual))
ggplot(train_na_filled, aes(x = TotRmsAbvGrd, y = logSalePrice)) +
  geom_jitter(aes(col = log(LotArea)), width = .3) + 
  scale_x_continuous(breaks = 2:14) +
  scale_colour_gradient(low = "white", high = "black")

## ----visualize bedroom related variables---------------------------------
ggplot(train_na_filled) + 
  geom_jitter(aes(x = BedroomAbvGr, y = logSalePrice, col = log(LotArea)), 
    width = .3, alpha = I(1/2)) +
  scale_x_continuous(breaks = 0:8) +
  scale_colour_gradient(low = "white", high = "black")

## ----visualize fireplaces------------------------------------------------
ggplot(train_na_filled) +
  geom_boxplot(aes(x = factor(Fireplaces), y = logSalePrice,
    col = FireplaceQu))

## ----visualize electrical and heating------------------------------------
pHeat <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = Heating, y = logSalePrice, 
    col = HeatingQC))
pElecAir <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = Electrical, y = logSalePrice,
    col = CentralAir))
grid.arrange(pHeat, pElecAir, nrow = 2)
pGrLiv <- ggplot(train_na_filled, aes(x = log(GrLivArea), y = logSalePrice)) + 
  geom_point(alpha = I(1/5)) + 
  geom_smooth(method = 'lm', level = .9)
pGrLivLot <- ggplot(train_na_filled) + 
  geom_point(aes(x = GrLivArea, y = LotArea), alpha = I(1/5)) + 
  scale_x_log10() + 
  scale_y_log10()
grid.arrange(pGrLiv, pGrLivLot, ncol = 2, widths = c(2, 3))

## ----visualize exterior variables----------------------------------------
ggplot(train_na_filled) + 
  geom_point(aes(x = LotFrontage, y = logSalePrice), alpha = I(1/5)) + 
  facet_wrap(~ MSSubClass, ncol = 5)
pMSSubClass <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = factor(MSSubClass), y = logSalePrice)) +
  theme(axis.text.x = element_blank())
pOverallQu <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = factor(OverallQual), y = logSalePrice)) +
  theme(axis.text.x = element_blank())
pOverallConv <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = factor(OverallCond), y = logSalePrice)) +
  theme(axis.text.x = element_blank())
pRfStl <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = RoofStyle, y = logSalePrice)) +
  theme(axis.text.x = element_blank())
pHsStl <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = HouseStyle, y = logSalePrice)) +
  theme(axis.text.x = element_blank())
pExQu <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = ExterQual, y = logSalePrice)) +
  theme(axis.text.x = element_blank())
pExCond <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = ExterCond, y = logSalePrice)) +
  theme(axis.text.x = element_blank())
pEx1 <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = Exterior1st, y = logSalePrice)) +
  theme(axis.text.x = element_blank())
pEx2 <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = Exterior2nd, y = logSalePrice)) +
  theme(axis.text.x = element_blank())
pLdConto <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = LandContour, y = logSalePrice)) +
  theme(axis.text.x = element_blank())
pLdSlp <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = LandSlope, y = logSalePrice)) +
  theme(axis.text.x = element_blank())
pBldgTp <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = BldgType, y = logSalePrice)) +
  theme(axis.text.x = element_blank())
grid.arrange(pMSSubClass, pOverallQu, pOverallConv, pRfStl,
             pHsStl, pExQu, pExCond, pLdConto, pLdSlp, pBldgTp, pEx1,
             pEx2, ncol = 4)

## ----visualize location and transporation--------------------------------
pMSzoning <- ggplot(train_na_filled) +
  geom_boxplot(aes(x = MSZoning, y = logSalePrice))
pNeighbor <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = Neighborhood, y = logSalePrice)) + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 5.5)) + 
  ylab('')
pStreet <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = Street, y = logSalePrice)) + 
  ylab('')
pPDrive <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = PavedDrive, y = logSalePrice)) + 
  ylab('')
pCond1 <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = Condition1, y = logSalePrice)) +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))
pCond2 <- ggplot(train_na_filled) + 
  geom_boxplot(aes(x = Condition2, y = logSalePrice)) +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) + 
  ylab('')
grid.arrange(pMSzoning, pStreet, pPDrive,
             pCond1, pCond2, pNeighbor, nrow = 2,
             heights = c(2, 3))

## ----visualize garage_---------------------------------------------------
pGargTp <- ggplot(train_na_filled) +
  geom_boxplot(aes(x = GarageType, y = logSalePrice)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 
pGargYBt <- ggplot(train_na_filled, aes(x = GarageYrBlt, y = logSalePrice)) + 
  geom_point(alpha = I(1/5), size = .5) +
  geom_smooth(method = 'lm') +
  ylab('')
pGargFns <- ggplot(train_na_filled, aes(x = GarageFinish, y = logSalePrice)) +
  geom_jitter(alpha = I(1/5), size = .5) 
pGargCars <- ggplot(train_na_filled, aes(x = GarageCars, y = logSalePrice)) + 
  geom_jitter(alpha = I(1/5), size = .5) +
  ylab('')
pGargAr <- ggplot(train_na_filled %>% filter(GarageArea > 0), 
       aes(x = GarageArea, y = logSalePrice)) + 
  geom_point(alpha = I(1/5), size = .5) + 
  geom_smooth(method = 'lm') 
pGargQu <- ggplot(train_na_filled, aes(x = GarageQual, y = logSalePrice)) + 
  geom_boxplot() +
  ylab('')
pGargCond <- ggplot(train_na_filled, aes(x = GarageCond, y = logSalePrice)) + 
  geom_boxplot() +
  ylab('')
grid.arrange(pGargAr, pGargYBt, pGargQu,
             pGargFns, pGargCars,
             pGargCond, pGargTp, 
             nrow = 3, heights = c(3, 3, 4),
             widths = c(3, 3, 2))

## ----visualize time related variables------------------------------------
pYrBlt <- ggplot(train_na_filled) +
  geom_point(aes(x = YearBuilt, y = logSalePrice), alpha = I(1/10))
pHsAge <- ggplot(train_na_filled) +
  geom_point(aes(x = YrSold - YearBuilt, y = logSalePrice), alpha = I(1/10))
pRemd <- ggplot(train_na_filled, aes(x = YrSold - YearRemodAdd, y = logSalePrice)) + 
  geom_point(alpha = I(1/10)) 
grid.arrange(pYrBlt, pHsAge, pRemd, nrow = 3, heights = c(2, 2, 2))

## ----visualize time related variables cond-------------------------------
ggplot(train_na_filled) +
  geom_boxplot(aes(x = factor(MoSold), y = logSalePrice)) + 
  facet_wrap(~ YrSold)
train_na_filled %>%
  group_by(MoSold, YrSold) %>%
  summarise(mid_logSalePrice = median(logSalePrice)) %>% 
  ggplot(aes(x = MoSold, y = mid_logSalePrice)) +
  geom_line(aes(group = YrSold, 
    col = factor(YrSold),
    linetype = factor(YrSold))) +
  geom_point(aes(group = YrSold, 
    col = factor(YrSold),
    linetype = factor(YrSold))) +
  scale_x_continuous(breaks = 1:12)

## ----visualize saletype salecondition and mosold-------------------------
options(digits = 2)
# prop of different SaleType above SaleCondition
table(train_na_filled$SaleType, train_na_filled$SaleCondition) / 
  rowSums(table(train_na_filled$SaleType, train_na_filled$SaleCondition))
# prop of different MoSold above SaleCondition
table(train_na_filled$MoSold, train_na_filled$SaleCondition) /
  rowSums(table(train_na_filled$MoSold, train_na_filled$SaleCondition))

## ----explore log-transf--------------------------------------------------
with(train_na_filled, 
  c(cor(logSalePrice, LotArea), 
    cor(logSalePrice, log(LotArea))))
with(train_na_filled %>% filter(TotalBsmtSF > 0), 
  c(cor(logSalePrice, TotalBsmtSF), 
    cor(logSalePrice, log(TotalBsmtSF))))
with(train_na_filled, 
  c(cor(logSalePrice, GrLivArea), 
    cor(logSalePrice, log(GrLivArea))))

## ----take log-transf-----------------------------------------------------
train_na_filled <- train_na_filled %>%
  mutate(
    logLotArea = log(LotArea),
    logTotalBsmtSF = log(TotalBsmtSF + 1),
    logGrLivArea = log(GrLivArea)
  )
test_na_filled <- test_na_filled %>%
  mutate(
    logLotArea = log(LotArea),
    logTotalBsmtSF = log(TotalBsmtSF + 1),
    logGrLivArea = log(GrLivArea)
  )

## ----add two binary variables--------------------------------------------
train_na_filled <- train_na_filled %>%
  mutate(
    hasBsmt = TotalBsmtSF > 0,
    hasGarg = GarageArea > 0)
test_na_filled <- test_na_filled %>%
  mutate(
    hasBsmt = TotalBsmtSF > 0,
    hasGarg = GarageArea > 0)

## ----saving the results--------------------------------------------------
save.image("C:/Users/Bangda/Desktop/kaggle/housing-price0806/eda.RData")
#save.image("C:/Users/bs2996/Downloads/housing-price0805/eda.RData")

