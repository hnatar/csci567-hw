import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert(len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels)+1

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return
        
    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred

    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')
        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent+'}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label # majority of current node

        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None # the dim of feature to be splitted
        self.feature_uniq_split = None # the feature to be splitted
        self.already_used = []


    def split(self):
        def conditional_entropy(branches: List[List[int]]) -> float:
            '''
            branches: C x B array, 
                      C is the number of classes,
                      B is the number of branches
                      it stores the number of 
            '''
            b=np.array(branches)
            bt = np.transpose(b).tolist()
            totsum = float(np.sum(b))
            ce = 0.0
            for branch in bt:
                bsum = float(sum(branch))
                if bsum == 0:
                    continue
                for class_label in branch:
                    if class_label > 0:
                        pi = float(class_label)/bsum
                        ce -= pi*np.log(pi)*(bsum/totsum)
            return ce

        """ If node is not splittable, just return """
        if not self.splittable:
            return

        attr = np.array(self.features).transpose()
        min_entropy = None

        """ Find best feature to split on """
        for idx_dim in range(len(self.features[0])):
            attr_unique = np.unique(attr[idx_dim])
            if len(attr_unique) == 1:
                """ This feature takes only one value in this node, cant use to split """
                continue
            for unique_attr_value in attr_unique:
                features_np = np.array(self.features)
                labels_np = np.array(self.labels)
                filtered_features=features_np[np.where(features_np[:, self.dim_split]==unique_attr_value)]
                filtered_labels=labels_np[np.where(features_np[:, self.dim_split]==unique_attr_value)]
                branches = np.zeros(shape = (self.num_cls,len(attr_unique)) )
                for i in range(len(self.features)):
                    index_of_value = attr_unique.tolist().index(self.features[i][idx_dim])
                    branches[self.labels[i]][index_of_value] += 1
                ce_temp = conditional_entropy(branches)
                if min_entropy == None or ce_temp <= min_entropy:
                    min_entropy = ce_temp
                    self.dim_split = idx_dim
                    self.feature_uniq_split = attr_unique.tolist()

        """ Find child splits for chosen feature """
        for unique_attr_value in self.feature_uniq_split:
            fcopy = [x for x in self.features]
            lcopy = [x for x in self.labels]
            features_np = np.array(self.features)
            filtered_features = features_np[np.where(features_np[:, self.dim_split]==unique_attr_value)]
            filtered_features = np.delete(filtered_features, self.dim_split, 1)
            labels_np = np.array(self.labels)
            filtered_labels=labels_np[np.where(features_np[:, self.dim_split]==unique_attr_value)]
            filtered_classes = len(np.unique(filtered_labels))



            tmp = TreeNode( filtered_features.tolist(), filtered_labels.tolist(), max(filtered_labels)+1)

            count_max = 0
            for label in np.unique(tmp.labels):
                if tmp.labels.count(label) > count_max:
                    count_max = tmp.labels.count(label)
                tmp.cls_max = label # majority of current node

            print('parent features=\n', self.features)
            print('parent labels=\n', self.labels)
            print('child features=\n', tmp.features)
            print('child labels=\n', tmp.labels)
            self.children.append(tmp)

        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()
        return

    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            # print(feature)
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max



