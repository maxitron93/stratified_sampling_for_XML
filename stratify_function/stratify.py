import random
import numpy as np
from datetime import datetime
import helper_funcs

def stratified_train_test_split(X, y, target_test_size, random_state=None):

    if random_state != None:
        random.seed(random_state)

    # To keep track of how long the initialization takes
    start_time = datetime.now()
    
    # Keep track how how many instances have been swapped to train or test
    swap_counter = {
        'to_train': 0,
        'to_test': 0,
    }

    # 1. Create instances_dict to keep track of instance information:
    # labels: array of labels, []
    # train_or_test: string, 'train' or 'test'
    # instance_score: float, adjusted sum of label scores
    instances_dict = helper_funcs.create_instances_dict(X, y, target_test_size)

    # 1.5 Get average number of labels per instance
    labels_per_instance = []
    for instance_id, instance_dict in instances_dict.items():
        labels_count = len(instance_dict['labels'])
        labels_per_instance.append(labels_count)
    average_labels_per_instance = sum(labels_per_instance) / len(labels_per_instance)

    # 2. Create labels_dict to keep track of label information:
    # train: int, number of times label appears in train set
    # test: int, number of times label appears in test set 
    # label_score: float, label score
    labels_dict = helper_funcs.create_labels_dict(instances_dict)

    # 3. Calculate the label score for each label in labels_dict
    # Positive score if too much of the label is in the test set
    # Negative score if too much of the label is in the train set
    helper_funcs.score_labels(labels_dict, target_test_size, average_labels_per_instance)

    # 4. Calculate the instance score for each instance in instances_dict
    # A high score means the instance is a good candidate for swapping
    helper_funcs.score_instances(instances_dict, labels_dict)

    # 5. Calculate the total score
    # The higher the score, the more 'imbalanced' the distribution of labels between train and test sets
    total_score = helper_funcs.calculate_total_score(instances_dict)
    print(f'Starting score: {round(total_score)}. Calculated in {str(datetime.now() - start_time).split(".")[0]}')

    # Main loop to create stratified train-test split  
    for i in range(50):
        
        # To keep track of how long each itteration takes
        itteration_start_time = datetime.now()
        
        # 6. Calculate the threshold score for swapping
        threshold_score = helper_funcs.calculte_threshold_score(instances_dict, average_labels_per_instance, i)

        # 7. Swap the instances with instance_score that is greater than the threshold score
        # Probability of swapping an instance is swap_probability
        helper_funcs.swap_instances(instances_dict, threshold_score, swap_counter, target_test_size, average_labels_per_instance, i)

        # 2. Recreate labels_dict with updated train-test split
        labels_dict = helper_funcs.create_labels_dict(instances_dict)

        # 3. Recalculate the label score for each label in labels_dict
        helper_funcs.score_labels(labels_dict, target_test_size, average_labels_per_instance)

        # 4. Recalculate the instance score for each instance in instances_dict
        helper_funcs.score_instances(instances_dict, labels_dict)

        # 5. Recalculate the total score
        total_score = helper_funcs.calculate_total_score(instances_dict)
        print(f'Epoch {i + 1}/50 score: {round(total_score)}. Calculated in {str(datetime.now() - itteration_start_time).split(".")[0]}')

    # Prepare X_train, X_test, y_train, y_test
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for instance_id, instance_dict in instances_dict.items():
        if instance_dict['train_or_test'] == 'train':
            X_train.append(X[instance_id])
            y_train.append(y[instance_id])
        elif instance_dict['train_or_test'] == 'test':
            X_test.append(X[instance_id])
            y_test.append(y[instance_id])
        else:
            print(f'Something went wrong: {instance_id}')
    
    # Print some statistics
    actual_test_size = len(X_test) / (len(X_train) + len(X_test))
    print(f'To train: {swap_counter["to_train"]}')
    print(f'To test: {swap_counter["to_test"]}')
    print(f'Target test size: {target_test_size}')
    print(f'Actual test size: {actual_test_size}')
    
    return X_train, X_test, y_train, y_test

