# loading the data --------------------------------------------------------

credit_data <- read.csv(file = "data/raw/creditcard.csv")


# import necessary libraries ----------------------------------------------

packages <- c("randomForest",
              "caret",
              "gridExtra",
              "grid",
              "ggplot2",
              "lattice",
              "corrplot",
              "pROC",
              "kableExtra",
              "formattable",
              "dplyr",
              "Rtsne",
              "data.table",
              "magrittr",
              "ggplot2",
              "plotly",
              "ggthemes")

check.packages <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

check.packages(packages)


# data preparation for visualization --------------------------------------

## Clean and select data --------------------------------------------------

data <- credit_data

data %>%
  mutate(id = 1:nrow(data)) %>%
  mutate(Class = as.integer(Class))

names(data) <- gsub('V', 'Feat', names(data))

numeric_interesting_features <- c(paste0('Feat', 1:28),
                                  'Amount') 

# "Class", the target, is not used to compute the 2D coordinates

data <- data[ apply(data, 
                    MARGIN = 1, 
                    FUN = function(x) !any(is.na(x))), ]


# Create normalized dataset of features -----------------------------------

df <- (as.data.frame(data[numeric_interesting_features]))

# "Class", the target, is not used to compute the 2D coordinates

df_normalised <- apply(df, 
                       MARGIN = 2, 
                       FUN = function(x) {
                         scale(x, center = T, scale = T)
                       })
df_normalised %<>%
  as.data.frame() %>%
  cbind(select(data, id))

# Remove line with potential NA
df_normalised <- df_normalised[ apply(df_normalised, MARGIN = 1, FUN = function(x) !any(is.na(x))), ]

data_fraud <- df_normalised %>%
  semi_join(filter(data, Class == 1), by = 'id')

data_sub <- df_normalised %>%
  sample_n(20000) %>% # sample of data
  rbind(data_fraud)

data_sub <- data_sub[!duplicated(select(data_sub, -id)), ]  
# remove rows containing duplicate values within rounding


# Run t-SNE to get the 2D coordinates -------------------------------------

rtsne_out <- Rtsne(as.matrix(select(data_sub, -id)), 
                   pca = FALSE, 
                   verbose = TRUE,
                   theta = 0.3, 
                   max_iter = 1300, 
                   Y_init = NULL)
# "Class", the target, is not used to compute the 2D coordinates


# Data post-processing ----------------------------------------------------

# merge 2D coordinates with original features
tsne_coord <- as.data.frame(rtsne_out$Y) %>%
  cbind(select(data_sub, id)) %>%
  left_join(data, by = 'id')


# Plot the map and its hexagonal background -------------------------------

gg <- ggplot() +
  labs(title = "All Frauds (white dots) in the transaction landscape (10% of data)") +
  scale_fill_gradient(low = 'darkblue', 
                      high = 'red', 
                      name="Proportion\nof fraud per\nhexagon") +
  coord_fixed(ratio = 1) +
  theme_void() +
  stat_summary_hex(data = tsne_coord, 
                   aes(x = V1, y = V2, z = Class), 
                   bins=10, 
                   fun = mean, 
                   alpha = 0.9) +
  geom_point(data = filter(tsne_coord, Class == 0), 
             aes(x = V1, y = V2), 
             alpha = 0.3, 
             size = 1, 
             col = 'black') +
  geom_point(data = filter(tsne_coord, Class == 1), 
             aes(x = V1, y = V2), 
             alpha = 0.9, 
             size = 0.3, 
             col = 'white') +
  theme(plot.title = element_text(hjust = 0.5, 
                                  family = 'Calibri'),
        legend.title.align = 0.5)

gg
#On about 10% of the data

# The hexagons show the local density of fraudulent transactions (white points). 
# Red colors mean high density of fraud (typically > 75% of points included in the hexagon) 
# whereas blueish colors are associated with a small fraction of fraud. 


# User defined functions --------------------------------------------------

# calculate ROC -----------------------------------------------------------

calculate_roc <- function(verset, cost_of_fp, cost_of_fn, n=100) {
  
  tp <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$Class == 1)
  }
  
  fp <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$Class == 0)
  }
  
  tn <- function(verset, threshold) {
    sum(verset$predicted < threshold & verset$Class == 0)
  }
  
  fn <- function(verset, threshold) {
    sum(verset$predicted < threshold & verset$Class == 1)
  }
  
  tpr <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$Class == 1) / sum(verset$Class == 1)
  }
  
  fpr <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$Class == 0) / sum(verset$Class == 0)
  }
  
  cost <- function(verset, threshold, cost_of_fp, cost_of_fn) {
    sum(verset$predicted >= threshold & verset$Class == 0) * cost_of_fp + 
      sum(verset$predicted < threshold & verset$Class == 1) * cost_of_fn
  }
  fpr <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$Class == 0) / sum(verset$Class == 0)
  }
  
  threshold_round <- function(value, threshold)
  {
    return (as.integer(!(value < threshold)))
  }
  
  # calculate AUC
  auc_ <- function(verset, threshold) {
    auc(verset$Class, threshold_round(verset$predicted,threshold))
  }
  
  roc <- data.frame(threshold = seq(0,1,length.out=n), tpr=NA, fpr=NA)
  roc$tp <- sapply(roc$threshold, function(th) tp(verset, th))
  roc$fp <- sapply(roc$threshold, function(th) fp(verset, th))
  roc$tn <- sapply(roc$threshold, function(th) tn(verset, th))
  roc$fn <- sapply(roc$threshold, function(th) fn(verset, th))
  roc$tpr <- sapply(roc$threshold, function(th) tpr(verset, th))
  roc$fpr <- sapply(roc$threshold, function(th) fpr(verset, th))
  roc$cost <- sapply(roc$threshold, function(th) cost(verset, th, cost_of_fp, cost_of_fn))
  roc$auc <-  sapply(roc$threshold, function(th) auc_(verset, th))
  
  return(roc)
}


# graphical representation for ROC, AUC and cost function -----------------

plot_roc <- function(roc, threshold, cost_of_fp, cost_of_fn) {
  library(gridExtra)
  
  norm_vec <- function(v) (v - min(v))/diff(range(v))
  
  idx_threshold = which.min(abs(roc$threshold-threshold))
  
  col_ramp <- colorRampPalette(c("green", "orange", "red", "black"))(100)
  
  col_by_cost <- col_ramp[ceiling(norm_vec(roc$cost) * 99) + 1]
  
  p_roc <- ggplot(roc, aes(fpr, tpr)) +
    geom_line(color = rgb(0, 0, 1, alpha = 0.3)) +
    geom_point(color = col_by_cost,
               size = 2,
               alpha = 0.5) +
    labs(title = sprintf("ROC")) + xlab("FPR") + ylab("TPR") +
    geom_hline(yintercept = roc[idx_threshold, "tpr"],
               alpha = 0.5,
               linetype = "dashed") +
    geom_vline(xintercept = roc[idx_threshold, "fpr"],
               alpha = 0.5,
               linetype = "dashed")
  
  p_auc <- ggplot(roc, aes(threshold, auc)) +
    geom_line(color = rgb(0, 0, 1, alpha = 0.3)) +
    geom_point(color = col_by_cost,
               size = 2,
               alpha = 0.5) +
    labs(title = sprintf("AUC")) +
    geom_vline(xintercept = threshold,
               alpha = 0.5,
               linetype = "dashed")
  
  p_cost <- ggplot(roc, aes(threshold, cost)) +
    geom_line(color = rgb(0, 0, 1, alpha = 0.3)) +
    geom_point(color = col_by_cost,
               size = 2,
               alpha = 0.5) +
    labs(title = sprintf("cost function")) +
    geom_vline(xintercept = threshold,
               alpha = 0.5,
               linetype = "dashed")
  
  sub_title <- sprintf("threshold at %.2f - cost of FP = %d, cost of FN = %d", threshold, cost_of_fp, cost_of_fn)
  
  grid.arrange(
    p_roc,
    p_auc,
    p_cost,
    ncol = 2,
    sub = textGrob(sub_title, gp = gpar(cex = 1), just = "bottom")
  )
}


# function for showing the confusion matrix -------------------------------

plot_confusion_matrix <- function(verset, sSubtitle) {
  tst <- data.frame(round(verset$predicted,0), verset$Class)
  opts <-  c("Predicted", "True")
  names(tst) <- opts
  cf <- plyr::count(tst)
  cf[opts][cf[opts]==0] <- "Not Fraud"
  cf[opts][cf[opts]==1] <- "Fraud"
  
  ggplot(data =  cf, mapping = aes(x = True, y = Predicted)) +
    labs(title = "Confusion matrix", subtitle = sSubtitle) +
    geom_tile(aes(fill = freq), colour = "grey") +
    geom_text(aes(label = sprintf("%1.0f", freq)), vjust = 1) +
    scale_fill_gradient(low = "lightblue", high = "blue") +
    theme_bw() +
    theme(legend.position = "none")
  
}


# exploring the data ------------------------------------------------------

nrow(credit_data)

ncol(credit_data)

summary(credit_data)

str(credit_data)

head(credit_data, 10) %>%
  kable( "html", 
         escape=F, 
         align="c") %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = F, 
                position = "center")

boxplot(credit_data$Amount)
hist(credit_data$Amount)

# There are totally 31 columns in the data. One column, `Class` is the target value; it is a binary value, can
# have either `0` (not fraud) or `1` (fraud) value. Another two columns have clear meaning: `Amount` is the
# amount of the transaction; `Time` is the time of the transaction. The rest of the features (28), anonymized, are
# named from `V1` to `V28`.
# The data is highly unbalanced with respect of `Class` variable values. There are only
# ``r nrow(credit_data[credit_data$Class==1,])/nrow(credit_data)*100`` % of the rows with value `Class = 1`. 
# Typically, in such cases, we can either choose to preserve the data unbalancing or use a oversampling 
# (of the data with minority value of target variable) or undersampling (of the data with majority value of the target variable).
# Here we will just preserve the unbalancing of the data. In terms of validation of the result, we will see that
# usual metrix, using a confusion matrix or accuracy are not the most relevant and will be prefered alternative
# solutions using AUC.


# correlations ------------------------------------------------------------

correlations <- cor(credit_data, method = "pearson")
corrplot(
  correlations,
  number.cex = .9,
  method = "circle",
  type = "full",
  tl.cex = 0.8,
  tl.col = "black"
)

# We can observe that most of the data features are not correlated. This is because before publishing,
# most of the features were presented to a Principal Component Analysis (PCA) algorithm.
# The features `V1` to `V28` are most probably the Principal Components resulted after propagating
# the real features through PCA. We do not know if the numbering of the features reflects the importance
# of the Principal Components. This information might be checked partially using the Variable Importance
# from Random Forest.


# Model -------------------------------------------------------------------

# After we split the data in a training and test set, we create the RF model using the training set.

nrows <- nrow(credit_data)
set.seed(314)
indexT <- sample(1:nrow(credit_data), 0.7 * nrows)

#separate train and validation set

trainset = credit_data[indexT, ]
verset = credit_data[-indexT, ]

n <- names(trainset)
rf.form <- as.formula(paste("Class ~", paste(n[!n %in% "Class"], collapse = " + ")))

trainset.rf <- randomForest(rf.form, 
                            trainset, 
                            ntree = 100, 
                            importance = T)


# visualizing the variable importance -------------------------------------

varimp <- data.frame(trainset.rf$importance)

vi1 <- ggplot(varimp, aes(x = reorder(rownames(varimp), IncNodePurity), y = IncNodePurity)) +
  geom_bar(stat = "identity",
           fill = "tomato",
           colour = "black") +
  coord_flip() + 
  theme_bw(base_size = 8) +
  labs(title = "Prediction using RandomForest with 100 trees",
       subtitle = "Variable importance (IncNodePurity)",
       x = "Variable",
       y = "Variable importance (IncNodePurity)")

vi2 <- ggplot(varimp, aes(x = reorder(rownames(varimp), X.IncMSE), y = X.IncMSE)) +
  geom_bar(stat = "identity",
           fill = "lightblue",
           colour = "black") +
  coord_flip() + theme_bw(base_size = 8) +
  labs(title = "Prediction using RandomForest with 100 trees",
       subtitle = "Variable importance (%IncMSE)",
       x = "Variable",
       y = "Variable importance (%IncMSE)")

grid.arrange(vi1, vi2, ncol = 2)


# Prediction --------------------------------------------------------------

# Let's use the trained model for prediction of the Fraud/Not Fraud Class for the test set.

verset$predicted <- predict(trainset.rf ,verset)

# For the threshold at 0.5, let's represent the Confusion matrix.

plot_confusion_matrix(verset, "Random Forest with 100 trees")

# For such a problem, where the number of TP is very small in comparison with the number of TN, 
# the Confusion Matrix is less useful, 
# since it is important to use a metric that include evaluation of FP and FN as well. 
# It is important to minimize as much as possible the number of FN (Predicted: Not Fraud and True: Fraud) 
# since their cost could be very large. Tipically AUC is used for such cases.

# Let's calculate the TP, FP, TN, FN, ROC, AUC and cost for threshold with values
# between 0 and 1 (100 values equaly distributed) and cost 1 for TN and 10 for FN.

roc <- calculate_roc(verset, 1, 10, n = 100)

mincost <- min(roc$cost)

roc %>%
  mutate(auc = ifelse(cost == mincost,
                      cell_spec(sprintf("%.5f", auc),
                                "html",
                                color = "green",
                                background = "lightblue",
                                bold = T),
                      cell_spec(sprintf("%.5f", auc),
                                "html",
                                color = "black",
                                bold = F))) %>%
  kable("html", escape = F, align = "c") %>%
  kable_styling(bootstrap_options = "striped",
                full_width = F,
                position = "center") %>%
  scroll_box(height = "600px")


# Let's plot the ROC, AUC and cost functions for a ref. threshold of 0.3.

threshold = 0.3
plot_roc(roc, threshold, 1, 10)


# Conclusion --------------------------------------------------------------

# The calculated accuracy is not very relevant in the conditions where there is a very large unbalance between
# the number of `fraud` and `non-fraud` events in the dataset. In such cases, we can see a very large accuracy.
# More relevant is the value of ROC-AUC (Area Under Curve for the Receiver Operator Characteristic). The value
# obtained (0.93) is relativelly good, considering that we did not performed any tunning, working with default
# RF algorithm parameters.