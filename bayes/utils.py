from csv import reader
import random

###### UTILS WILL HAVE TO MOVE FROM HERE
def split_data(data, split):
    random.seed(a=None)
    train_split = int(len(data)*split)
    train = []
    for i in range(train_split):
        random_index = random.randrange(len(data))
        train.append(data[random_index])
        data.pop(random_index)
    return [train, data]

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def	s_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def s_to_int(dataset, column):
	class_ids = [row[column] for row in dataset]
	unique = set(class_ids)
	lookup = {}
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

def label_feature_split_back(train, test, class_pos):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i in range(len(train)):
        Y_train.append(train[i][class_pos-1])
        X_train.append(train[i][:class_pos-1])
            
    for i in range(len(test)):
        Y_test.append(test[i][class_pos-1])
        X_test.append(test[i][:class_pos-1])
    return X_train, X_test, Y_train, Y_test

def label_feature_split_front(train, test, class_pos):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i in range(len(train)):
        Y_train.append(train[i][class_pos-1])
        X_train.append(train[i][class_pos:])
            
    for i in range(len(test)):
        Y_test.append(test[i][class_pos-1])
        X_test.append(test[i][class_pos:])
    return X_train, X_test, Y_train, Y_test
