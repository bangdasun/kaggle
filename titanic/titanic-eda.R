## ---- include=FALSE------------------------------------------------------
knitr::opts_chunk$set(warning = FALSE, message = FALSE)

## ------------------------------------------------------------------------
setwd('C:/Users/Bangda/Desktop/kaggle/titanic')
library(ggplot2)
library(reshape2)
library(stringr)
library(dplyr)
library(rpart)
library(e1071)
train <- read.csv('train.csv', header = TRUE)
test  <- read.csv('test.csv', header = TRUE)

## ------------------------------------------------------------------------
glimpse(train)

## ------------------------------------------------------------------------
glimpse(test)

## ------------------------------------------------------------------------
train$Family <- ifelse((train$SibSp + train$Parch) == 0, 0, 1)
test$Family  <- ifelse((test$SibSp + test$Parch) == 0, 0, 1)
train$num_family <- train$SibSp + train$Parch
test$num_family  <- test$SibSp + test$Parch

## ------------------------------------------------------------------------
head(train$Name)

## ------------------------------------------------------------------------
extract_name <- function(name, type) {
  # split Name to title, first and last name
  splited_name <- str_split(name, '[,.]') %>% unlist() %>% str_trim()
  return(splited_name[type])
}
# add new variables to train and test
train$first_name <- sapply(train$Name, extract_name, 1)
train$title      <- sapply(train$Name, extract_name, 2)
train$last_name  <- sapply(train$Name, extract_name, 3)
test$first_name  <- sapply(test$Name, extract_name, 1)
test$title       <- sapply(test$Name, extract_name, 2)
test$last_name   <- sapply(test$Name, extract_name, 3)
# check the categories of title
train$title %>% unique()
test$title  %>% unique()
# get the list of all categories
titles <- base::union(train$title %>% unique(), test$title %>% unique())
# factorize title
train$title <- train$title %>% factor(levels = titles)
test$title  <- test$title %>% factor(levels = titles)

## ------------------------------------------------------------------------
apply(train, 2, function(x) sum(is.na(x)) )
apply(test,  2, function(x) sum(is.na(x)) )

## ------------------------------------------------------------------------
idx_na_fare <- which(is.na(test$Fare))
# with sex
test[-idx_na_fare, ] %>%
  ggplot() + 
  geom_boxplot(aes(x = Sex, y = Fare)) 
# with pclass
test[-idx_na_fare, ] %>%
  ggplot() +
  geom_boxplot(aes(x = factor(Pclass), y = Fare)) + 
  facet_wrap(~ Sex)
# with embarked
test[-idx_na_fare, ] %>%
  ggplot() + 
  geom_boxplot(aes(x = Embarked, y = Fare)) + 
  facet_grid(Sex ~ Pclass)

## ------------------------------------------------------------------------
test[idx_na_fare, c('Sex', 'Pclass', 'Embarked')]
# use mean and median
test[-idx_na_fare, ] %>%
  select(Sex, Pclass, Embarked, Fare) %>%
  filter(Sex == 'male', Pclass == 3, Embarked == 'S') %>%
  summarise(avg_fare = mean(Fare), mid_fare = median(Fare))

## ------------------------------------------------------------------------
test[idx_na_fare, 'Fare'] <- 7.9875

## ------------------------------------------------------------------------
idx_na_embark <- ((train$Embarked %>% as.character()) == '') %>% which()
train[idx_na_embark, ] %>%
  select(Survived, Fare, Sex, Pclass)
# with fare, embarked, pclass
train[-idx_na_embark, ] %>%
  ggplot() +
  geom_boxplot(aes(x = Embarked, y = Fare, col = Sex)) + 
  facet_grid(Survived ~ factor(Pclass))

## ------------------------------------------------------------------------
train[idx_na_embark, 'Embarked'] <- 'C'
train$Embarked <- factor(train$Embarked, levels = c('C', 'Q', 'S'))

## ------------------------------------------------------------------------
idx_na_age_train <- which(is.na(train$Age))
train_wna_age  <- train[idx_na_age_train, ]
train_wona_age <- train[-idx_na_age_train, ]
# age ~ sex | survived
train_wona_age %>%
  ggplot() +
  geom_boxplot(aes(x = factor(Survived), y = Age, col = Sex)) + 
  xlab('Survived')
# age ~ sex | pclass
train_wona_age %>%
  ggplot() + 
  geom_boxplot(aes(x = factor(Pclass), y = Age, col = Sex)) +
  xlab('Pclass')
# age ~ title
train_wona_age %>%
  ggplot() +
  geom_boxplot(aes(x = title, y = Age)) + 
  facet_wrap(~ Sex, nrow = 2)
# age ~ family | survived
train_wona_age %>%
  ggplot() +
  geom_boxplot(aes(x = factor(Family), y = Age, col = factor(Survived))) +
  xlab('Family')
# age ~ numfamily
train_wona_age %>%
  filter(num_family > 0) %>%
  ggplot() +
  geom_boxplot(aes(x = factor(num_family), y = Age)) + 
  xlab('Number of family members')
# age ~ fare | pclass + sex
train_wona_age %>%
  ggplot() +
  geom_point(aes(x = Fare, y = Age), alpha = I(1/5)) +
  facet_grid(Sex ~ Embarked, scale = 'free')

## ------------------------------------------------------------------------
# train and validation set
train_age <- train_wona_age %>%
  dplyr::select(PassengerId, Age, Pclass, Fare, Family, num_family, title) %>%
  dplyr::mutate(Pclass = factor(Pclass), Family = factor(Family))
# need to fill age
pred_age  <- train_wna_age %>%
  dplyr::select(PassengerId, Age, Pclass, Fare, Family, num_family, title) %>%
  dplyr::mutate(Pclass = factor(Pclass), Family = factor(Family))
# test set
test <- test %>%
  dplyr::mutate(Pclass = factor(Pclass), Family = factor(Family))
set.seed(123)
train_train_idx <- sample(1:nrow(train_age), round(nrow(train_age)/5), replace = FALSE)
train_train_age <- train_age[train_train_idx, ]
train_test_age  <- train_age[-train_train_idx, ]

## ------------------------------------------------------------------------
# training model
tr_model1 <- rpart(Age ~., data = train_train_age)
# training error
pred_tr_model1_train <- predict(tr_model1, newdata = train_train_age)
(rmse_tr_model1_train <- (pred_tr_model1_train - train_train_age$Age)^2 %>% mean() %>% sqrt())
# test error
pred_tr_model1_test <- predict(tr_model1, newdata = train_test_age)
(rmse_tr_model1_test <- (pred_tr_model1_test - train_test_age$Age)^2 %>% mean() %>% sqrt())

## ------------------------------------------------------------------------
# tunning model
tr_model2 <- tune.rpart(Age ~., data = train_train_age, maxdepth = 2:7, minsplit = 2:10, 
    cp = c(0.001, 0.002, 0.005, 0.01, 0.02, 0.03))
tr_model2$best.parameters
# relative best model
tr_model3 <- rpart(Age ~., data = rbind(train_train_age, train_test_age), 
    maxdepth = 5, minsplit = 5, cp = .005)
# training error
pred_tr_model3_train <- predict(tr_model3, newdata = train_train_age)
(rmse_tr_model3_train <- (pred_tr_model3_train - train_train_age$Age)^2 %>% mean() %>% sqrt())
# test error
pred_tr_model3_test <- predict(tr_model3, newdata = train_test_age)
(rmse_tr_model3_test <- (pred_tr_model3_test - train_test_age$Age)^2 %>% mean() %>% sqrt())

## ------------------------------------------------------------------------
pred_age$Age <- predict(tr_model3, newdata = pred_age)
# fill in train
train_wna_age <- train_wna_age %>%
  left_join(pred_age[, c('PassengerId', 'Age')], by = 'PassengerId') %>%
  select(-Age.x) %>%
  mutate(Age = Age.y) %>%
  select(-Age.y)
train <- rbind(train_wna_age, train_wona_age)
# fill in test
test_wna_age  <- test[is.na(test$Age), ]
test_wona_age <- test[!is.na(test$Age), ]
test_wna_age$Age <- predict(tr_model3, newdata = test_wna_age)
test <- rbind(test_wna_age, test_wona_age)
# save the data
save.image("C:/Users/Bangda/Desktop/kaggle/titanic/eda1.RData")

## ------------------------------------------------------------------------
train$adult <- ifelse(train$Age < 18, 0, 1) %>% factor()
test$adult  <- ifelse(test$Age  < 18, 0, 1) %>% factor()
# survived ~ adult | sex
train %>%
  ggplot(aes(x = adult, fill = factor(Survived), 
    col = factor(Survived))) +
  geom_bar(position = 'fill') +
  facet_wrap(~ Sex) + ylab('prop')

## ------------------------------------------------------------------------
# distribution of age
train %>%
  ggplot(aes(x = Age, fill = Sex)) +
  geom_histogram(aes(y = ..density..), 
                 col = 'black', alpha = I(1/3), bins = 40)
train %>%
  ggplot(aes(x = Age, fill = Sex)) + 
  geom_density(aes(y = ..density..), alpha = I(1/3))

## ------------------------------------------------------------------------
# survival at different age 
train %>%
  ggplot() +
  geom_freqpoly(aes(x = Age, col = factor(Survived)), bins = 60, size = 1.5)
# survival at different age | sex
train %>%
  ggplot() +
  geom_freqpoly(aes(x = Age, col = factor(Survived)), bins = 60, size = 1.5) + 
  facet_wrap(~ Sex, nrow = 2, scale = 'free_y')

## ------------------------------------------------------------------------
train$old <- ifelse(train$Age < 70, 0, 1) %>% factor()
test$old  <- ifelse(test$Age  < 70, 0, 1) %>% factor()
# survived ~ old | sex
train %>%
  ggplot(aes(x = old, fill = factor(Survived), col = factor(Survived))) +
  geom_bar(position = 'fill') +
  facet_wrap(~ Sex) + ylab('prop')

## ------------------------------------------------------------------------
all_ticket <- union(unique(train$Ticket) %>% as.character(), unique(test$Ticket) %>% as.character())
c(length(all_ticket), nrow(train) + nrow(test))

## ------------------------------------------------------------------------
# extract letters
extract_letters <- function(ticket) {
  letter <- str_extract_all(ticket, '[A-Z]+') %>% '[[' (1)
  return(paste(letter, collapse = ''))
}
# extract numbers
extract_numbers <- function(ticket) {
  number <- str_extract_all(ticket, '[0-9]+') %>% '[[' (1) 
  return(nchar(paste(number, collapse = '')))
}
# extract
letter <- sapply(c(train$Ticket %>% as.character(), test$Ticket %>% as.character()), extract_letters)
number <- sapply(c(train$Ticket %>% as.character(), test$Ticket %>% as.character()), extract_numbers)
re_ticket <- paste0(letter, number)
# add re_ticket into data
train$re_ticket <- re_ticket[1:nrow(train)]
test$re_ticket <- re_ticket[(nrow(train) + 1):(nrow(test) + nrow(train))]
level <- union(train$re_ticket, test$re_ticket)
train$re_ticket <- factor(train$re_ticket, levels = level)
test$re_ticket <- factor(test$re_ticket, levels = level)
# check the survival rate of different type of ticket
ggplot(train, aes(x = re_ticket, fill = factor(Survived))) +
  geom_bar(aes(col = factor(Survived)),width = 0.5, position = 'fill') +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) 
ggplot(train, aes(x = re_ticket, fill = factor(Survived))) +
  geom_bar(aes(col = factor(Survived)),width = 0.5, position = 'fill') +
  facet_wrap(~ Pclass, nrow = 1) +
  coord_flip()

## ------------------------------------------------------------------------
idx_na_cabin <- which(train$Cabin == '')
train[-idx_na_cabin, ] %>%
  mutate(layer = str_sub(Cabin, 1, 1) %>% factor) %>%
  ggplot(aes(x = layer, fill = factor(Survived))) +
  geom_bar(aes(col = factor(Survived)), position = 'fill') 

## ------------------------------------------------------------------------
idx_family_train <- which(train$Family == 1) 
idx_family_test  <- which(test$Family == 1) 
train_family <- train[idx_family_train, ]
test_family  <- test[idx_family_test, ]
train_nfamily <- train[-idx_family_train, ]
test_nfamily  <- test[-idx_family_test, ]

## ------------------------------------------------------------------------
train_nfamily$FamilyId <- 0
test_nfamily$FamilyId  <- 0

## ------------------------------------------------------------------------
family_key <- union(train_family$first_name,
                    test_family$first_name)
family_ref <- train_family %>%
  select(first_name, num_family) %>%
  rbind(test_family[, c('first_name', 'num_family')]) %>%
  group_by(first_name) %>%
  summarise(count = n())
family_ref$FamilyId <- 1:nrow(family_ref)
train_family <- train_family %>%
  left_join(family_ref[, c('first_name', 'FamilyId')], by = 'first_name')
test_family  <- test_family %>%
  left_join(family_ref[, c('first_name', 'FamilyId')], by = 'first_name')
# concatenate back
train <- rbind(train_nfamily, train_family)
test  <- rbind(test_nfamily, test_family)

## ------------------------------------------------------------------------
save.image("C:/Users/Bangda/Desktop/kaggle/titanic/eda2.RData")

## ------------------------------------------------------------------------
glimpse(train)

