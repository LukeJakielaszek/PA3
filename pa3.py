'''
Add the code below the TO-DO statements to finish the assignment. Keep the interfaces
of the provided functions unchanged. Change the returned values of these functions
so that they are consistent with the assignment instructions. Include additional import
statements and functions if necessary.
'''

import csv
import numpy as np
import matplotlib.pyplot as plt

# return the average logistic loss over all samples
def logistic_loss(train_y, pred_y):
    # get the prediction
    prediction = train_y * pred_y

    #calculate element wise logistic loss
    elem_log = np.log(1+np.exp(-prediction))

    # calculate the average of the logistic losses
    log_loss = np.mean(elem_log)

    # return the average log_loss
    return log_loss

# return the average hinge loss over all samples
def hinge_loss(train_y, pred_y):
    # get the prediction
    prediction = train_y * pred_y    

    # calculate element wise hinge loss
    elem_hinge = np.maximum(0, 1-prediction)
    
    # calculate the average hinge loss
    hinge_loss = np.mean(elem_hinge)

    # return the average hinge loss
    return hinge_loss

'''
The regularizers shall compute the loss without considering the bias term in the weights
'''
# compute the l1 norma of the weight vector
def l1_reg(w):
    # computes the l1 norm without while ignoring the bias term
    return np.linalg.norm(w[:-1], ord=1)

# compute the  l2 norm of the weight vector
def l2_reg(w):
    # computes the l2 norm without while ignoring the bias term
    return np.linalg.norm(w[:-1], ord=2)

def train_classifier(train_x, train_y, learn_rate, loss, lambda_val=None, regularizer=None):
    # generate our weight vector drawn from normal distribution in the scale of .01
    w = np.random.normal(scale=.01, size = len(train_x[0]))

    # number of iterations
    num_iters = 30000

    # run through all iterations
    for i in range(num_iters):
        # calculate teh numerical gradient
        gradient = calculate_gradient(w, train_x, train_y, loss, lambda_val, regularizer)

        # update w according to gradient
        w = w - learn_rate * gradient

    # return the learned weights
    return w

def calculate_gradient(w, train_x, train_y, loss, lambda_val=None, regularizer=None):
    # get a prediction for each sample
    pred_y = np.dot(train_x, w)

    # compute loss without h
    loss_val = loss(train_y, pred_y)

    # calculate regularizer without h
    reg_val = 0
    if(lambda_val):
        reg_val = lambda_val*regularizer(w)
    
    # define a small h for numerical gradient calculation
    h = .00001
    
    # holds per feature loss when adding h to each feature
    loss_with_h = np.zeros(len(w))

    # holds per feature regularizer when adding h to each feature
    reg_with_h = np.zeros(len(w))

    for col in range(len(w)):
        # add h to the current feature
        w[col] += h

        # get a prediction for each sample using modified weight vector
        pred_y = np.dot(train_x, w)

        # compute loss with h
        loss_with_h[col] = loss(train_y, pred_y)

        # compute regularizer with h
        if(lambda_val):
            reg_with_h[col] = lambda_val*regularizer(w)
        
        # remove h from the current feature
        w[col] -= h

    # calculate objective function without h
    objective_base = reg_val + loss_val

    # calculate objective function with h
    objective_h = np.zeros(len(w))
    for i in range(len(w)):
        objective_h[i] = reg_with_h[i] + loss_with_h[i]

    # initialize a gradient vector
    gradient = np.zeros(len(w))

    # calculate the per feature numerical gradient
    for i in range(len(w)):
        gradient[i] = (objective_h[i] - objective_base) / h

    # return the gradient vector
    return gradient

# get predictions for each sample
def test_classifier(w, test_x):
    # return predictions for each sample
    return np.dot(test_x, w)

# performs n-fold cross validation
def n_fold_cross_validate(dataX, dataY, n, learn_rate, loss, lambda_val=None, regularizer=None):
    # get a permutation object to randomize the data
    perm = np.random.permutation(len(dataY))

    # randomize the data
    dataX = dataX[perm]
    dataY = dataY[perm]    

    # split the data into n sections
    split_dataX = np.split(dataX, n)
    split_dataY = np.split(dataY, n)

    # store average accuracy
    average_acc = 0
    
    # run each cross validation experiment
    for i in range(n):
        # obtain validation block data
        test_x = split_dataX[i]
        test_y = split_dataY[i]

        # Reset training variable and initialize it to empty np array
        train_x = np.array([])
        train_y = np.array([])

        # loop through all blocks
        for index in range(n):
            # concatenate the block as training data if it is not currently in our testing set
            if(index != i):
                # initializes train_x and train_y to first valid block
                if(not train_x.any()):
                    train_x = split_dataX[i]
                    train_y = split_dataY[i]
                else:
                    # concatenate additional blocks
                    train_x = np.concatenate((train_x, split_dataX[i]))
                    train_y = np.concatenate((train_y, split_dataY[i]))

        # compute the normalization parameters from the training data
        mean, std = compute_norm_params(train_x)
        
        #normalize the training data using training normalization params
        # (also append bias vector)
        train_x = normalize_data(train_x, mean, std)

        # normalize the validation data using training normalization params
        # (also append bias vector)
        test_x = normalize_data(test_x, mean, std)

        # train the model on the current training set
        w = train_classifier(train_x, train_y, learn_rate, loss, lambda_val, regularizer)

        # get predictions on the validation set from the trained SVM
        pred_y = test_classifier(w, test_x)

        # compute the accuracy for each prediction on the validation set
        accuracy = compute_accuracy(test_y, pred_y)

        average_acc += accuracy
        print("\t\tAccuracy: " + str(accuracy))

    # compute the average accuracy
    average_acc /= n

    # display the average
    print("\tAverage Accuracy: " + str(average_acc))

# computes accuracy by comparing predictions to known values
def compute_accuracy(test_y, pred_y):
    # count correctly identified classes
    num_correct = 0.0

    # loop through the dataset
    for i in range(len(test_y)):
        prediction = 0
        # determine what class the prediction is
        if(pred_y[i] > 0):
            prediction = 1
        else:
            prediction = -1
            
        # if a prediction was correct, increment the count
        if(test_y[i] == prediction):
            num_correct += 1.0

    # return the percent of correctly classified data
    return num_correct / len(test_y)
        
# normalizes the dataset by subtracting mean and dividing by standard deviation
def normalize_data(data_x, mean, std):
    # transpose the data to work with feature vectors
    data_x = np.transpose(data_x)

    # loop through each feature
    for i in range(len(data_x)):
        # subtract the mean from each entry
        data_x[i] = data_x[i] - mean[i]

        # divide each entry by the supplied standard deviation
        data_x[i] = data_x[i] / std[i]

    # undo the transpose to have feature columns
    data_x = np.transpose(data_x)

    # create the bias vector
    bias = np.ones((len(data_x),1))

    # append the bias vector to the matrix
    data_x = np.append(data_x, bias, axis=1)
    
    # return the normalized data
    return data_x
    
# compute the mean and standard deviation for every feature in our data        
def compute_norm_params(data):
    # transpose the data to work with features as rows
    data_transpose = np.transpose(data)

    # initialize our per feature mean and standard deviation to all zeros
    mean = np.zeros([len(data_transpose),])
    std = np.zeros([len(data_transpose),])

    # loop through each feature
    for i, feat in enumerate(data_transpose):
        # calculate the feature's mean
        mean[i] = np.mean(feat)

        # calculate the feature's standard deviation
        std[i] = np.std(feat)

    # return the mean and standard deviations of all features
    return mean, std

def main():

    # Read the training data file
    szDatasetPath = 'winequality-white.csv'
    listClasses = []
    listAttrs = []
    bFirstRow = True
    with open(szDatasetPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            if bFirstRow:
                bFirstRow = False
                continue
            if int(row[-1]) < 6:
                listClasses.append(-1)
                listAttrs.append(list(map(float, row[1:len(row) - 1])))
            elif int(row[-1]) > 6:
                listClasses.append(+1)
                listAttrs.append(list(map(float, row[1:len(row) - 1])))

    dataX = np.array(listAttrs)
    dataY = np.array(listClasses)

    # 5-fold cross-validation
	# Note: in this assignment, preprocessing the feature values will make
	# a big difference on the accuracy. Perform feature normalization after
	# spliting the data to training and validation set. The statistics for
	# normalization would be computed on the training set and applied on
	# training and validation set afterwards.
	# TO-DO: Add your code here

    # learning rates for parameter sweep
    learning_rates = [.1, .01, .001, .0001, .00001]

    # regularization lambdas for parameter sweep
    lambdas = [100, 10, 1, .1, .01, .001]

    # perform a perameter sweep of all values for SVM
    print("SVM (Hinge Loss with L2 Regularizer):")

    # loop through each learning rate
    for learning_rate in learning_rates:
        # loop through each lambda
        for lambda_val in lambdas:
            print("\tLearning Rate: " + str(learning_rate))
            print("\tLambda: " + str(lambda_val))
            
            # perform 5-fold cross validation using the lr and lambda for SVM
            n_fold_cross_validate(dataX, dataY, 5, learning_rate, hinge_loss, lambda_val, l2_reg)

    print("---------------------------------------------------")
    print("---------------------------------------------------")

    print("Logistic Regression (Logistic Loss without Regularizer):")
    # perform a perameter sweep of all values for logistic regression
    # loop through each learning rate
    for learning_rate in learning_rates:
        print("\tLearning Rate: " + str(learning_rate))

        # perform 5-fold cross validation using the lr and lambda for logistic regression
        n_fold_cross_validate(dataX, dataY, 5, learning_rate, logistic_loss)

    return None

if __name__ == "__main__":

    main()
