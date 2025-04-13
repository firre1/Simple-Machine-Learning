#!/usr/bin/env python
# coding: utf-8

# # ID3 implementation

# Some required imports.
# Make sure you have these packages installed on your system.
import pandas as pd
import numpy as np
import math 
from sklearn.metrics import confusion_matrix, accuracy_score

# Global variable for node ids
node_id_counter = 1

# Some functions needed when creating the decision tree
class Utils:
    
    # Calculates the entropy for the values in target_column, which should be of type Pandas.DataFrame
    @staticmethod
    def entropy(target_column):
        
        num_instances_all_classes = len(target_column)
        
        value_counts = target_column.value_counts()

        entropy = 0
        
        # value_counts might be indexed by single value (e.g. 'Yes') or a tuple, depending on how Pandas is handling it
        for target_value in value_counts.keys():
            # If it's a tuple, extract the actual value
            if isinstance(target_value, tuple):
                target_value = target_value[0]

            num_instances_current_class = value_counts[target_value]
            share_instances_current_class = num_instances_current_class / num_instances_all_classes
            entropy = entropy - share_instances_current_class * math.log2(share_instances_current_class)

        return entropy
    
    # Identifies the dominating class for the entries in target_column
    # target_column should be of type Pandas.DataFrame 
    @staticmethod
    def find_dominating_class(target_column):
        return_class = None
        count_return_class = 0
        
        value_counts = target_column.value_counts()
        for target_value in value_counts.keys():
            if isinstance(target_value, tuple):
                candidate_class = target_value[0]
            else:
                candidate_class = target_value
            
            if value_counts[target_value] > count_return_class:
                count_return_class = value_counts[target_value]
                return_class = candidate_class
        
        return return_class
    
    # This method returns a unique node identifier
    @staticmethod
    def get_next_node_id():
        global node_id_counter
        node_id = node_id_counter
        node_id_counter += 1
        return node_id
        
    # Resets the global variable node_counter_id used to assign unique identifiers for the nodes.
    @staticmethod
    def reset_node_id_counter():
        global node_id_counter
        node_id_counter = 1
    

# This class is used to represent a node in the decision tree. 
class Node:
    
    # Constructor for the node class taking the following parameters.
    def __init__(self, height, max_height, input_columns, target_column, parent, parent_split_value):
        
        # Height in the tree for the current node. 
        # It is suggested that the height of the root of the tree is 1. 
        self.height = height
        
        # The maximum height of the decision tree, meaning that no path from the root 
        # to a leaf should contain more than 3 nodes. 
        self.max_height = max_height
        
        # Pandas columns containing the input features. 
        # One column for each input feature, and each row is data point (or instance)
        self.input_columns = input_columns
        
        # Pandas column containing the target variable. 
        # Each row in this column contains the target value corresponding to the same row in self.input_columns 
        self.target_column = target_column
        
        # The variable to split on in this node. 
        # For leaf nodes this is not set.
        self.split_variable = None
        
        # The names of the input features
        self.input_names = self.input_columns.keys()
        
        # The name of the target variable
        self.target_name = self.target_column.keys()[0]
        
        # Reference to the parent of this node. 
        # This is None for the root of the tree
        self.parent = parent
        
        # The class for this node if it is a leaf node. 
        # This is left empty for non leaf nodes. 
        self.class_name = None
        
        # The unique id of this node. Used when printing the tree. 
        self.id = Utils.get_next_node_id()
    
        # Dictionary used to keep track of the child nodes. Empty if the node is a leaf. 
        self.children = {}

        # Value of the parent's split leading here (useful for printing)
        self.parent_split_value = parent_split_value

    # Method is used to expand a node when constructing the decision tree  
    def split(self):
        # 1. Check if leaf (all target values the same, or at max height)
        unique_targets = self.target_column[self.target_name].unique()
        if len(unique_targets) == 1:
            self.class_name = unique_targets[0]
            return
        
        if self.height >= self.max_height:
            # Make it a leaf using dominating class
            dominating_class = Utils.find_dominating_class(self.target_column[self.target_name])
            self.class_name = dominating_class
            return

        # 2. Compute information gain for each attribute, pick the best
        current_entropy = Utils.entropy(self.target_column[self.target_name])
        best_gain = -1
        best_attribute = None
        
        for attr in self.input_names:
            # For each possible value of attr, split
            values = self.input_columns[attr].unique()
            split_entropy = 0

            for val in values:
                mask = (self.input_columns[attr] == val)
                subset = self.target_column[self.target_name][mask]
                if len(subset) == 0:
                    continue
                prob = len(subset) / len(self.target_column)
                split_entropy += prob * Utils.entropy(subset)
            
            info_gain = current_entropy - split_entropy
            if info_gain > best_gain:
                best_gain = info_gain
                best_attribute = attr

        if best_attribute is None:
            # No good split found, so leaf
            dominating_class = Utils.find_dominating_class(self.target_column[self.target_name])
            self.class_name = dominating_class
            return

        # Store best attribute as the split variable
        self.split_variable = best_attribute

        # 3. Create children for each distinct value of the best_attribute
        unique_values = self.input_columns[best_attribute].unique()
        for val in unique_values:
            mask = (self.input_columns[best_attribute] == val)

            child_input_columns = self.input_columns[mask].drop(columns=[best_attribute])
            child_target_column = self.target_column[mask]

            child_node = Node(
                height = self.height + 1,
                max_height = self.max_height,
                input_columns = child_input_columns,
                target_column = child_target_column,
                parent = self,
                parent_split_value = val
            )
            # Recursively split
            child_node.split()

            # Add to children dict
            self.children[val] = child_node

        # If no children ended up being created (edge case), become leaf
        if len(self.children) == 0:
            dominating_class = Utils.find_dominating_class(self.target_column[self.target_name])
            self.class_name = dominating_class

    # This method creates a text representation of the decision tree.   
    def print(self):
        print("Node<" + str(self.id) + ">" )
        
        if not self.children:
            # Leaf
            if self.parent is not None:
                print("  Leaf node - Parent: " + str(self.parent.id) + 
                      ", SplitValue from parent: " + str(self.parent_split_value) +
                      ", Decision: " + self.class_name)
            else:
                print("  Leaf node - Parent: None, Decision: " + self.class_name)
        else:
            # Non-leaf
            if self.parent is None:
                print("  Non leaf node - Parent: None")
            else:
                print("  Non leaf node - Parent: " + str(self.parent.id) +
                      ", SplitValue from parent: " + str(self.parent_split_value))
            print("  Split variable: " + self.split_variable)
            
            for child_split_value in self.children.keys():
                child_node = self.children[child_split_value]  
                print("    Child_node: " + str(child_node.id) + 
                      ", split_value: " + str(child_split_value))

# Class used to represent our decision tree generated using the ID3 algorithm
class DecisionTree:

    # Constructor that takes three input variables:
    #   max_height - The maximum height of the decision tree. 
    #   instances - Pandas DataFrame respresenting instance vectors (both input and target)
    #   target_name - Name of the target column.
    def __init__(self, max_height, instances, target_name):
        
        # Reset the global node_id_counter variable to make sure the root node gets id 1
        Utils.reset_node_id_counter()
        
        self.max_height = max_height
        
        # Create pandas dataframe containing only the target column. 
        self.target_column = instances[[target_name]]
 
        if self.target_column[target_name].unique().size != 2:
            print("Error: Only binary target variables are supported")
            exit()
        
        # Create pandas dataframe containing all input feature columns.
        self.input_columns = instances.drop([target_name], axis=1) 
      
        # Create the root of the tree
        self.root = Node(1, self.max_height, self.input_columns, self.target_column, None, None)

        # Generate the decision tree by calling the self.generate() function.
        self.generate()
    
    # This method is used to generate a decision tree using the ID3 algorithm
    # All tree nodes are generated recursively when adding new nodes.
    def generate(self):
        self.root.split()
    
    # This method prints all nodes in the decision tree
    def print_tree(self):
        if self.root is None:
            print("Decision tree is emtpy")
            return
        
        node_queue = []
        node_queue.append(self.root)
        
        while node_queue:
            current_node = node_queue.pop(0)
            current_node.print()
            
            for child_node in current_node.children.values():
                node_queue.append(child_node)


def predict_instance(decision_tree, instance):
    node = decision_tree.root
    
    while node.class_name is None:
        split_var = node.split_variable
        val = instance[split_var]
        
        if val not in node.children:
            return Utils.find_dominating_class(node.target_column[node.target_name])
        
        node = node.children[val]
        
    return node.class_name


# This function contains the code to create your decision tree 
def main():
    # Read the data from csv file and store in a pandas DataFrame
    golf_dataframe = pd.read_csv("golf_dataset.csv")

    print(golf_dataframe)
    print("\n")

    max_height = 3
    
    # Generate decision tree for golf data set, target variable "Play Golf" and max_height=3 
    dt = DecisionTree(max_height, golf_dataframe, 'Play Golf')

    # Print content of the created tree
    dt.print_tree()

    # Confusion Matrix
    X = golf_dataframe.drop('Play Golf', axis=1)
    y_true = golf_dataframe['Play Golf']
    y_pred = []

    for i in range(len(X)):
        instance = X.iloc[i]
        pred_class = predict_instance(dt, instance)
        y_pred.append(pred_class)

    cm = confusion_matrix(y_true, y_pred, labels=["No", "Yes"])
    accuracy = accuracy_score(y_true, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No", "Actual: Yes"],
        columns=["Predicted: No", "Predicted: Yes"]
    )

    print("\nConfusion Matrix:")
    print(cm_df)
    print(f"\nAccuracy on the same dataset: {accuracy:.2f}")

# Call the main function to run the code
main()
