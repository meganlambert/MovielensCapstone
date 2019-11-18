################################
# Create edx set, validation set
################################
# Note: this process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Examine the first rows of the edx dataset with headers
head(edx)

#View the summary statistics for the edx dataset
summary(edx)

#Number of unique users and movies
edx %>% summarize(n_users=n_distinct(userId), n_movies=n_distinct(movieId))

#Number of rows in edx dataset
nrow(edx)

#Visualize the fact that some movies get rated more times than others
edx %>%     dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

#Visualize the fact that some users rate more movies than others. Most users have rated more than 30 movies.
edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

#Looking at how the number of ratings are distributed
edx %>% group_by(rating) %>% summarize(count = n())

#Function we will use to calculate the RMSE of each model
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Compute the average movie rating on the training data
mu_hat <- mean(edx$rating)
mu_hat

#Predict the unknown ratings to be mu and compute the residual mean squared error on the test data.
naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse

#Create a results table to display the RMSE results of our various models
rmse_results <- tibble(method = "Mean Rating Only", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

#Define the overall movie rating mean
mu <- mean(edx$rating) 

#Compute the average rating for each movie
movie_average_rating <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating-mu))

#Plot the average movie rating
movie_average_rating %>% 
  qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

#Predict the ratings using the movie effect model
predicted_ratings_movie_effect <- mu  + validation %>% 
  left_join(movie_average_rating, by='movieId') %>%
  .$b_i

#Compute the RMSE of our new movie effect model
model_movie_effect_rmse <- RMSE(predicted_ratings_movie_effect, validation$rating)
rmse_results <- bind_rows(rmse_results, tibble(method="Movie Effect Model", RMSE = model_movie_effect_rmse))
rmse_results %>% knitr::kable()

#Plot user average rating
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

#Compute our approximation by computing the overall mean, u, the movie effects, b_i, and then estimating the user effects, b_u.
user_average_rating <- edx %>%
  left_join(movie_average_rating, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#Predict ratings using both movie and user effect
predicted_ratings_movie_and_user_effect <- validation %>%
  left_join(movie_average_rating, by='movieId') %>%
  left_join(user_average_rating, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

#Compute the RMSE of the movie and user effect model
model_movie_and_user_effect_rmse <- RMSE(validation$rating,predicted_ratings_movie_and_user_effect)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie and User Effect Model",  
                                     RMSE = model_movie_and_user_effect_rmse ))
rmse_results %>% knitr::kable()

#We apply a cross validation method to choose the best lamda.
lambdas <- seq(0, 10, 0.25)
#For each lamda we will calculate b_i and b_u to set our predictions and test them
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings_regularized <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings_regularized, validation$rating))
})

#Plot the lamdas and RMSEs
qplot(lambdas, rmses) 

#Find the optimal value for lamda
lambda <- lambdas[which.min(rmses)]
lambda

#Compute the RMSE for the regularized Moive and User effect model
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User Effect Model",
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

#Results
rmse_results %>% knitr::kable()

