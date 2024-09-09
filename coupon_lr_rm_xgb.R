#install packages
options(repos = c(CRAN = "https://cran.rstudio.com/"))
install.packages('tidyverse')
install.packages('lubridate')
install.packages('dplyr')
install.packages('tidyr')
install.packages('caret')
install.packages('ggplot2')
install.packages('ROSE')
install.packages('randomForest')
install.packages("xgboost")

#import packages
library(tidyverse)
library(lubridate)
library(ggplot2)
library(caret)

# Read the CSV file
data <- read_csv("offline_train.csv")

# Display the first 10 rows of the dataset
view(head(data, 10))
head(data, 10)

# Check for missing values in the Discount_rate column
table(is.na(data$Discount_rate))

# Display unique values in the Discount_rate column
unique(data$Discount_rate)

# Replace "null" with NA in the Discount_rate column
data$Discount_rate <- ifelse(data$Discount_rate == "null", NA, data$Discount_rate)

# Recheck for missing values in the Discount_rate column
table(is.na(data$Discount_rate))

# Recheck unique values in the Discount_rate column
unique(data$Discount_rate)

# Check for missing values in the Distance column
table(is.na(data$Distance))

# Display unique values in the Distance column
unique(data$Distance)

# Replace "null" with NA in the Distance column
data$Distance <- ifelse(data$Distance == "null", NA, data$Distance)

# Recheck unique values in the Distance column
unique(data$Distance)

# Convert the Distance column to numeric
data$Distance <- as.numeric(data$Distance)

# Plot a histogram of the Distance column
ggplot(data, aes(x = Distance)) +
  geom_histogram(bins = 20, color = "black", fill = "lightblue") +
  labs(x = "Distance", y = "Frequency", title = "Histogram of Distance")

# Calculate the median of the Distance column
median_distance <- median(data$Distance, na.rm = TRUE)

# Fill missing values in the Distance column with the median
data$Distance[is.na(data$Distance)] <- median_distance


# Check if the 'Date' and 'Date_received' columns are in date format
class(data$Date)
class(data$Date_received)

unique(data$Date)
unique(data$Date_received)

library(lubridate)

# Convert the Date and Date_received columns to date format 
data$Date <- ymd(data$Date)  
data$Date_received <- ymd(data$Date_received)  

# Check the conversion results for Date and Date_received
unique(data$Date)
unique(data$Date_received)

# Recheck if the 'Date' and 'Date_received' columns are in date format
class(data$Date)
class(data$Date_received)

# Replace "null" with NA in the Coupon_id column
data$Coupon_id <- ifelse(data$Coupon_id == "null", NA, data$Coupon_id)
#unique(data$Coupon_id)
table(is.na(data$Coupon_id))

# Check if both Coupon_id and Date_received columns are either both NA or both not NA
data$both_null <- is.na(data$Coupon_id) & is.na(data$Date_received)

# Count the rows where both columns are NA
sum(data$both_null)

# Count the rows where neither column is NA
sum(!data$both_null)

# Display the distribution of rows where both columns are NA or not
table(data$both_null)

# Identify rows where one column is NA and the other is not
data$one_null <- is.na(data$Coupon_id) != is.na(data$Date_received)

# Count the rows where one column is NA and the other is not
sum(data$one_null)

# Plot a bar chart showing the distribution of rows where both columns are NA or not

ggplot(data, aes(x = both_null)) +
  geom_bar(fill = "lightblue") +
  labs(x = "Both Columns Null", y = "Count", title = "Distribution of Null Values")



# Specify the date format for the Date and Date_received columns
data$Date <- as.Date(data$Date, format = "%Y-%m-%d")
data$Date_received <- as.Date(data$Date_received, format = "%Y-%m-%d")

# Remove rows where Coupon_id is NA
data <- data[!is.na(data$Coupon_id), ]
view(head(data))
head(data)

#unique(data$Coupon_id)

# Display unique values in the Discount_rate column
unique(data$Discount_rate)

# Check the class (data type) of the Discount_rate column
class(data$Discount_rate)



library(dplyr)
library(tidyr)
# Generate features
data <- data %>%
  mutate(Discount_rate = as.character(Discount_rate)) %>%  # 确保 Discount_rate 为字符型
  separate(Discount_rate, c("discount_threshold", "discount_amount"), sep = ":", convert = TRUE, fill = "right") %>%
  mutate(
    discount_threshold = ifelse(is.na(discount_threshold), 0, discount_threshold),  # 将 NA 填充为 0
    discount_amount = ifelse(is.na(discount_amount), 0, discount_amount),  # 将 NA 填充为 0
    new_discount_rate = ifelse(discount_threshold == 0, as.numeric(Discount_rate), 
                               1 - discount_amount / discount_threshold),  # 计算新的折扣率
    discount_strength = 1 / new_discount_rate  # 计算折扣力度
  )

# View the processed data
head(data)

# Check the transformation results
glimpse(data)

colnames(data)

# Determine if the coupon was used
data <- data %>%
  mutate(
    Coupon_used = ifelse(!is.na(Coupon_id) & !is.na(Date), "Yes", "No")
  )

colSums(is.na(data))

str(data)

# Check the ratio of positive and negative samples
table(data$Coupon_used)


# Select features
X <- data[, c("discount_threshold", "discount_amount", "Distance", "new_discount_rate", "discount_strength")]

# Select the target variable
y <- factor(data$Coupon_used)

head(X)
head(y)

glimpse(y)
glimpse(X)
table(y)
summary(X)
cor(X)

# Load machine learning packages
library(caret)
library(ROSE)
library(xgboost)
library(ggplot2)

# Data splitting
set.seed(42)
train_index <- createDataPartition(y, p = 0.75, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Use the ROSE package for a combination of oversampling and undersampling
balanced_data <- ROSE(Coupon_used ~ ., data = cbind(X_train, Coupon_used = y_train), seed = 42)$data
X_resampled <- balanced_data[, !(names(balanced_data) %in% c("Coupon_used"))]
y_resampled <- balanced_data$Coupon_used

# Logistic regression model
# Convert Coupon_used to factor and specify the reference level
balanced_data$Coupon_used <- factor(balanced_data$Coupon_used, levels = c("No", "Yes"))

# Train the logistic regression model
log_reg <- glm(Coupon_used ~ ., data = balanced_data, family = binomial)

# Make predictions on the test set
y_pred_log <- predict(log_reg, newdata = X_test, type = "response")

# Adjust the threshold according to specific needs, here using 0.5 as an example
y_pred_log_class <- ifelse(y_pred_log > 0.5, "Yes", "No")

# Evaluate the logistic regression model performance
confusion_matrix <- confusionMatrix(data = as.factor(y_pred_log_class), reference = as.factor(y_test), positive = "Yes")
print(confusion_matrix)

# Visualize the confusion matrix of logistic regression
cm_log <- confusionMatrix(factor(y_pred_log_class), factor(y_test))
ggplot(data = as.data.frame(cm_log$table),
       aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%d", Freq))) +
  scale_fill_gradient(low = "yellow", high = "purple") +
  theme_minimal() +
  labs(title = "Confusion Matrix - Logistic Regression")

# Random Forest model

# Load the randomForest library
library(randomForest)

# Train a Random Forest model
rf_model <- randomForest(x = X_resampled, 
                         y = as.factor(y_resampled), 
                         ntree = 50,  # 调整树的数量
                         mtry = sqrt(ncol(X_resampled)),  # 默认设置
                         importance = TRUE)


# Predict on the test set using the trained Random Forest model
rf_pred <- predict(rf_model, newdata = X_test)

# Evaluate the model performance using a confusion matrix
confusionMatrix(as.factor(rf_pred), as.factor(y_test))

# View feature importance from the Random Forest model
importance(rf_model)
varImpPlot(rf_model)

#XGBoost model

# Install and load the xgboost package
library(xgboost)

# Define model training control parameters

ctrl <- trainControl(
  method = "cv",
  number = 5,  # 5-fold cross-validation
  search = "grid",
  verboseIter = TRUE,
  allowParallel = TRUE
)


# Define the grid of hyperparameters for XGBoost
xgb_grid <- expand.grid(
  nrounds = c(50, 100), 
  max_depth = c(3, 6), 
  eta = c(0.1, 0.3),  
  gamma = 0, 
  colsample_bytree = 0.8, 
  min_child_weight = 1, 
  subsample = 0.8  
)


# Train the XGBoost model using the train function
xgb_model <- train(
  x = as.matrix(X_resampled),  # XGBoost需要矩阵输入
  y = y_resampled,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  objective = "binary:logistic",  # 对于二分类问题
  eval_metric = "auc",
  verbose = FALSE,
  nthread = parallel::detectCores() - 1  # 使用多线程但留一个核心给系统
)

# 加载必要的库
library(xgboost)

# Print the trained model and best hyperparameters
print(xgb_model)
print(xgb_model$bestTune)

# Make predictions on the test data
xgb_pred <- predict(xgb_model, newdata = as.matrix(X_test))
levels(xgb_pred)

# Optionally, convert predictions to numeric and display the first 10

xgb_pred_numeric <- predict(xgb_model, newdata = as.matrix(X_test))
head(xgb_pred_numeric,n=10)

# Convert numeric predictions to factors and calculate the confusion matrix

xgb_pred_factor <- factor(xgb_pred_numeric, levels = c("No", "Yes"))
cm <- table(Predicted = xgb_pred_factor, Actual = y_test)
print(cm)

# Calculate and display feature importance

importance <- varImp(xgb_model, scale = FALSE)
print(importance)
plot(importance)