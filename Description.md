## Preprocessing

I start by preprocessing the data, splitting it into training and test sets and scaling it. Due to the limited amount of training data I've also decided to drop some columns. The models perform better without them.

## Models

I've decided to use 4 different models: KNN, SVM, DT and RF. 

## Hyperparameter tuning 

The program tunes each model using RepeatedStratifiedKFold()

## Training and testing

The models are trained on the training dataset and then tested on the test dataset

## Output

The program outputs the mean accuracy scores and their standard deviations for each model as well as paramiters determined to be the best by the hyperparameter tuning functions.
