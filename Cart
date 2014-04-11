import numpy
import argparse
from sklearn import cross_validation


class Tree(object):
    """
    CART - Model to predict number of user's friends in VK
    """
    def __init__(self):
        self.answer = None
        self.type = None
        self.left = None
        self.right = None
        self.attribute = [0, 0, 0, 0]

    #X - [0]gender, [1]relationships, [2]wall, [3]subscriptions, [4]photos
    #type : 0 - list, 1 - node

    def sum(self, X, p):
        s = 0
        for i in range(0, len(X)):
            s += float(X[i][p])
        return s

    def create_list(self, X, y):
        self.type = 0
        self.attribute = [0, 0, 0, 0]
        self.answer = sum(y) / float(len(X))


    def is_list_criteria(self, depth):
        if depth < 3: # Maximum DEPTH
            return 0
        else:
            return 1


    #We divide all members into two classes
    #In first current attribute i < average of all, in second >=
    @staticmethod
    def gini_impurity(self, X, y, depth):
        if self.is_list_criteria(depth):
            self.create_list(X, y)
            return self
        self.type = 1
        self.attribute[3] = -1
        for i in range(0, len(X[0])):
            s = self.sum(X, i) / float(len(X))
            left = []
            lefty = []
            right = []
            righty = []
            for j in range(0, len(X)):
                if X[j][i] < s:
                    left.append(X[j])
                    lefty.append(y[j])
                else:
                    right.append(X[j])
                    righty.append(y[j])
            impurity = 1 - (len(right) ** 2 + len(left) ** 2) / float(len(X)) ** 2
            if len(right) < 2 or len(left) < 2:
                continue
            temp = Tree()
            temp.left = Tree()
            temp.right = Tree()
            temp.type = 1
            temp.attribute = [i, s, impurity, None]
            temp.left = temp.left.gini_impurity(temp.left, left, lefty, depth + 1)
            temp.right = temp.right.gini_impurity(temp.right, right, righty, depth + 1)
            temp.attribute[3] = impurity - len(right) / float(len(X)) * temp.right.attribute[2] - len(left) / float(len(X)) * temp.left.attribute[2]
            if temp.attribute[3] > self.attribute[3]:
                self = temp
        if self.attribute[3] == -1:
            self.create_list(X, y)
        return self

    def build_tree(self, X, y):
        self = self.gini_impurity(self, X, y, 0)
        return self

    def fit(self, X, y):
        t = self.build_tree(X, y)
        self.attribute = t.attribute
        self.answer = t.answer
        self.type = t.type
        self.left = t.left
        self.right = t.right

    def predict_user(self, user):
        if self.type:
            if user[self.attribute[0]] < self.attribute[1]:
                return self.left.predict_user(user)
            else:
                return self.right.predict_user(user)
        else:
            return self.answer

    def predict(self, X):
        res = []
        for i in range(0, len(X)):
            res.append(self.predict_user(X[i]))
        return res

    def get_params(self, deep = True):
        return {}

    def score(self, X, y):
        y_prediction = numpy.array(self.predict(X)) - numpy.array(y)
        return numpy.sqrt(numpy.dot(y_prediction, y_prediction)/len(y_prediction))


def readCSV(fname):
    f = open(fname, "r")
    x = []
    y = []
    f.readline();
    line = f.readline();
    while (line and len(line) > 1):
        try:
            tk = line.split(',')
            x.append([float(tk[3]), float(tk[4]), float(tk[6]), float(tk[7]), float(tk[8])])
            y.append(float(tk[-1]))
        except:
            tk = line.split('"')
            tk = tk[2].split(',')
            x.append([float(tk[1]), float(tk[2]), float(tk[4]), float(tk[5]), float(tk[6])])
            y.append(float(tk[-1]))
        line = f.readline()
    f.close()
    return x, y


def launch(flearn, ftest):
    x, y = readCSV(flearn)
    CART = Tree()
    CART.fit(x, numpy.array(y))
    print(CART.attribute[1])
    f = open(ftest, "r")
    f.readline()
    line = f.readline()
    sumsq = 0
    i = 0
    print("Prediction:\n")
    while (line and len(line) > 1):
        try:
            tk = line.split(',')
            t = [float(tk[3]), float(tk[4]), float(tk[6]), float(tk[7]), float(tk[8])]
            temp = CART.predict_user(t)
            print(str([tk + ' prediction is ' + temp]))
            sumsq += (float(tk[-1]) - temp) ** 2
            i += 1
            line = f.readline()
        except:
            tk = line.split('"')
            tk = tk[2].split(',')
            t = [float(tk[1]), float(tk[2]), float(tk[4]), float(tk[5]), float(tk[6])]
            temp = CART.predict_user(t)
            print(str([tk + ' prediction is ' + temp]))
            sumsq += (float(tk[-1]) - temp) ** 2
            i += 1
            line = f.readline()
    print("Standard deviation: " + str(numpy.sqrt(sumsq / i)))
    f.close()

def crossVal(fname):
    x, y = readCSV(fname)
    scores = cross_validation.cross_val_score(Tree(), x, numpy.array(y), cv=5)
    print("Standard deviation: " + str(sum(scores)/len(scores)))

def parse_args():
    parser = argparse.ArgumentParser(
        description="ClassificationAndRegressionTree for finding number of user's friends on VK")
    parser.add_argument('learn_path', nargs=1)
    parser.add_argument('test_path', nargs=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.learn_path[0] == 'cross-val':
        crossVal(args.test_path[0])
    else:
        launch(args.learn_path[0], args.test_path[0])

if __name__ == "__main__":
    main()
