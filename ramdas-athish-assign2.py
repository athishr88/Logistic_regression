import math
from math import log
import csv
import random
import numpy as np

save_to_csv = False
training_split = 0.8
testing = True


# Function to import reviews
def review_file_generator(filename_review):
    # Takes filename of the review and returns two lists, both of them in the format - one review per list entry
    # 1. List of reviews without special characters
    # 2. List of reviews with special characters
    with open(filename_review, "r", encoding="utf8") as file:
        review_sp = file.readlines()  # Text with special characters
        review_sp = [line.rstrip() for line in review_sp]

        review = [line.replace(",", " ").replace(".", " ").replace("!", " ").replace("(", " ").replace(")", " ") for
                  line in review_sp]

        return review, review_sp


# Function to generate list of vocabulary from text file
def lexicon_generator(filename_lexicon):
    # Takes file name and returns a list of words
    with open(filename_lexicon, "r", encoding="utf8") as file:
        words = file.readlines()
        words = [line.rstrip() for line in words]
        return words


# Function to extract features from review files
def feature_extractor(review, review_sp, positive_words, negative_words, pronoun_list, category=None):
    features = []
    i = 0
    for line in review:
        line_list = line.lower().split()
        x1 = 0
        x2 = 0
        x3 = 0
        x4 = 0
        x5 = 0
        x6 = 0
        ID = line_list[0]

        # Counting positive words in the review
        for pos_word in positive_words:
            x1 += line_list.count(pos_word.lower())

        # Counting negative words in the review
        for neg_word in negative_words:
            x2 += line_list.count(neg_word.lower())

        # Check for 'No' in the review
        if "no" in line_list:
            x3 = 1
        else:
            x3 = 0

        # Counting pronouns in the review
        for pronoun in pronoun_list:
            x4 += line_list.count(pronoun)

        # Check for '!' in the review
        if "!" in review_sp[i]:
            x5 = 1
        else:
            x5 = 0

        # Length of review
        x6 = math.log(len(line_list) - 1)

        features.append([ID.upper(), x1, x2, x3, x4, x5, x6, category])
        i += 1

    return features


# Function to calculate sigmoid
def sigmoid(number):
    sigmoid_out = (1/(1+math.exp(-number)))
    return sigmoid_out


if __name__ == '__main__':
    # region Feature Extraction

    hotel_pos_t, hotel_pos_t_sp = review_file_generator('hotelPosT-train.txt')
    hotel_neg_t, hotel_neg_t_sp = review_file_generator('hotelNegT-train.txt')
    pos_words = lexicon_generator('positive-words.txt')
    neg_words = lexicon_generator('negative-words.txt')
    pronouns = ["I", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"]

    # Generating features for positive reviews
    positive_features = feature_extractor(hotel_pos_t, hotel_pos_t_sp, pos_words, neg_words, pronouns, 1)

    # Generating features for negative reviews
    negative_features = feature_extractor(hotel_neg_t, hotel_neg_t_sp, pos_words, neg_words, pronouns, 0)

    # Concatenating all features
    all_features = positive_features + negative_features

    # Storing features in a csv file
    if save_to_csv:
        filename = "all_features.csv"
        with open(filename, "w", newline="") as csvfile:
            csvWriter = csv.writer(csvfile)

            csvWriter.writerows(all_features)

    # endregion Feature Extraction

    # region Logistic Regression (SGD)
    num_of_features = len(all_features)
    random.shuffle(all_features)

    # Training and validation data split
    train_data = all_features[0:math.floor(training_split*num_of_features)]
    validation_data = all_features[math.floor(training_split * num_of_features):]

    # Variables
    weights = np.array([0, 0, 0, 0, 0, 0, 0.1], dtype='float32')  # Initializing weights
    epochs = 100
    learning_rate = .01
    CE_agg = 0
    i = 0

    train_data_num = len(train_data)
    # correct_train = 0
    # Training loop
    for epoch in range(epochs):
        correct_train = 0
        # CE_agg = 0
        for data in train_data:
            temp = data[1:7] + [1]  # Adding the dummy feature for bias
            feature = np.array(temp, dtype='float32')

            y = float(data[-1]) # Ground truth
            y_dash = sigmoid(np.dot(feature, weights)) # Prediction

            # Cross entropy loss
            CE_loss = -math.log((y_dash**y)*(1-y_dash)**(1-y))

            y_dash_pred = float(round(y_dash))
            if y_dash_pred == y:
                correct_train += 1
            # Learning weights
            gradient = (y_dash - y)*feature
            weights = weights - (learning_rate * gradient)

            CE_agg += CE_loss
            i += 1

        average = CE_agg/i
        accuracy_train = correct_train/train_data_num
        # print("Cross-entropy loss - ", average, "Accuracy", accuracy_train)

    # Validation
    j = 0
    correct = 0
    for data in validation_data:
        temp = data[1:7] + [1]  # Adding the dummy feature for bias
        feature = np.array(temp, dtype='float32')

        y = data[-1]
        y_dash = sigmoid(np.dot(feature, weights))

        # Prediction
        y_dash = round(y_dash)
        if y_dash == y:
            correct += 1

        j += 1

    average = correct/j
    print("Rightly predicted: ",correct,"Total test set number", j, "Accuracy", average)

    # Testing
    if testing:
        # Reading the review
        test_set, test_set_s = review_file_generator("HW2-testset.txt")
        # Generating features
        test_features = feature_extractor(test_set, test_set_s, pos_words, neg_words, pronouns)

        output_file = open("ramdas-athish-assign2-out.txt", "w+")

        # Prediction loop
        for data in test_features:
            temp = data[1:7] + [1]  # Adding the dummy feature for bias
            feature = np.array(temp, dtype='float32')

            y_dash = sigmoid(np.dot(feature, weights))

            # Prediction
            y_dash = round(y_dash)

            # Writing the output file
            if y_dash == 0:
                output_file.write("%s\t NEG\n" % data[0])
            else:
                output_file.write("%s\t POS\n" % data[0])

        output_file.close()
        pass
