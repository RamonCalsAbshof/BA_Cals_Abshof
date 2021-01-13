from sklearn.tree import DecisionTreeClassifier
from numpy.random import seed

seed(1) # set seed to guarantee reproducibility


def fit_decision_tree(features, labels, min_samples_split=2, max_depth=None, class_weight='balanced', random_state=10):
    dtree = DecisionTreeClassifier(min_samples_split=min_samples_split,
                                   max_depth=max_depth,
                                   class_weight=class_weight,
                                   random_state=random_state).fit(features, labels)
    return dtree


def predict(dtree, features):
    return dtree.predict(features)


def eval(dtree, features, labels):
    return dtree.score(features, labels)


def run_decision_tree(train_X, train_Y, test_X, test_Y, min_samples_split=2, max_depth=None, class_weight=None,
                      random_state=1, print_tree=True):
    dtree = DecisionTreeClassifier(min_samples_split=min_samples_split,
                                   max_depth=max_depth,
                                   class_weight=class_weight,
                                   random_state=random_state).fit(train_X, train_Y)
    if print_tree:
        print_dtree(dtree)
    score = dtree.score(test_X, test_Y)
    return dtree, score


# print decision tree in console
def print_dtree(estimator):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            print(estimator.tree_.value[i])
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print()