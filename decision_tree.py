# LEARNING MONK-1 decision Tree using our own algo and calculating the confusion matrix and accuracy score
import numpy as np
import matplotlib.pyplot as plt
import math
def partition(x): 
  
   s={}
   for v in np.unique(x):
     s.update({v: (x == v).nonzero()[0]})
   return s

def entropy(y):
    values,counts=np.unique(y,return_counts=True)
    h=0
    for i in range(len(values)):
        p=counts[i]/len(y)
        h-=p*math.log2(p)
    return h
    raise Exception('Function not yet implemented!')

def mutual_information(x, y):
    
    hy=entropy(y)
    x_vals,counts=np.unique(x,return_counts=True)
    px= counts/len(x)
    mapped=zip(px,x_vals)
    for i,v in mapped:
      hy-=i*entropy(y[x==v])
    return hy
  
    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    tree = {}
    if attribute_value_pairs is None:
        attribute_value_pairs = np.vstack([[(i, v) for v in np.unique(x[:, i])] for i in range(x.shape[1])])
    y_values, y_counts = np.unique(y, return_counts=True)

    if len(y_values) == 1:
        return y_values[0]
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        return y_values[np.argmax(y_counts)]
    mutual_info = np.array([mutual_information(np.array(x[:, i] == v).astype(int), y)
                                 for (i, v) in attribute_value_pairs])
    (attr, value) = attribute_value_pairs[np.argmax(mutual_info)]
    partitions = partition(np.array(x[:, attr] == value).astype(int))
    attribute_value_pairs = np.delete(attribute_value_pairs, np.argwhere(np.all(attribute_value_pairs == (attr, value), axis=1)), 0)

    for split_value, indices in partitions.items():
        x_subset = x.take(indices, axis=0)
        y_subset = y.take(indices, axis=0)
        decision = bool(split_value)

        tree[(attr, value, decision)] = id3(x_subset, y_subset, attribute_value_pairs=attribute_value_pairs,
                                            max_depth=max_depth, depth=depth + 1)

    return tree


def predict_example(x, tree):
    for split_criterion, sub_trees in tree.items():
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if split_decision == (x[attribute_index] == attribute_value):
            if type(sub_trees) is dict:
                label = predict_example(x, sub_trees)
            else:
                label = sub_trees

            return label
    raise Exception('Could not predict example with this tree.')


def compute_error(y_true, y_pred):
    n = len(y_true)
    err = [y_true[i] != y_pred[i] for i in range(n)]
    return sum(err) / n


def pretty_print(tree, depth=0):
   
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def visualize(tree, dot_string='', parent='', nid=-1, depth=0):

    nid += 1

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if type(sub_trees) is dict:
            if split_decision:
                node_id = 'node{0}'.format(nid)
                dot_string, left_child, nid = visualize(sub_trees, dot_string=dot_string,
                                                        parent=node_id, nid=nid, depth=depth+1)
                dot_string += '    {0} -> {1} [label="True"];\n'.format(node_id, left_child)
            else:
                node_id = 'node{0}'.format(nid)
                dot_string += '    {0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)
                dot_string, right_child, nid = visualize(sub_trees, dot_string=dot_string,
                                                         parent=node_id, nid=nid, depth=depth + 1)
                dot_string += '    {0} -> {1} [label="False"];\n'.format(node_id, right_child)
        else:
            node_id = 'node{0}'.format(nid)
            dot_string += '    {0} [label="y = {1}"];\n'.format(node_id, sub_trees)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, nid

def confusionMatrixCalculation(p_labels,t_labels):
    
    pred_labels = np.asarray(p_labels)
    true_labels = np.asarray(t_labels)
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    print ('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))
    return [[TP, FN] , [FP, TN]]                      #Mainting the order as per scikit learns implementation


if __name__ == '__main__':
    #basepath=r'C:\Users\sxs180011\Downloads\data\data'
    for dataSetCounter in range(1,4):
        M = np.genfromtxt('./data/monks-'+str(dataSetCounter)+'.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]
        tst_Error_Array = []
        depth_Array = []
        tr_error_array=[]
        M = np.genfromtxt('./data/monks-'+str(dataSetCounter)+'.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]
    
        for tree_depth in range(1,11):
            print("Depth of the tree is currently :" + str(tree_depth))
            decision_tree = id3(Xtrn, ytrn, max_depth = tree_depth)
            pretty_print(decision_tree)
            dot = visualize(decision_tree)
            print(dot)
            y_pred = [predict_example(x, decision_tree) for x in Xtst]
            tst_err = compute_error(ytst, y_pred)        
            conf = confusionMatrixCalculation(y_pred,ytst)
            print(conf)
            plt.imshow(conf, cmap='binary', interpolation='None')
            plt.title('confusion matrix for Monks-1 at depth' + str(tree_depth) )
            plt.savefig('Monks__' + str(dataSetCounter)+'_confusionmatrix__'+ str(tree_depth) +'.png')
            plt.show()
            print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
            depth_Array.append(tree_depth)
            tst_Error_Array.append(tst_err)
            y_tr=[predict_example(x,decision_tree) for x in Xtrn]
            tr_error=compute_error(y_tr,ytrn)
            print('Training error={0:4.2f}%'.format(tr_error*100))
            tr_error_array.append(tr_error)
            #print(tr_error)
        fig,ax=plt.subplots()
        plt.xlabel('depth of tree')
        plt.ylabel(' error')
        plt.plot(depth_Array,tr_error_array,'r',label='training error')
        plt.plot(depth_Array, tst_Error_Array,'b',label='test error')
        legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
        plt.savefig('monks__'+ str(dataSetCounter)+ '.png')
        plt.show()   
