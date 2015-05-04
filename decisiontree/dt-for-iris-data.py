
## DecisionTreeClassifier for multi-classification


from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

from sklearn.externals.six import StringIO

with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

import pydot, sys
sys.path.append('C:\\Program Files (x86)\\Graphviz2.38\\bin')
# print(sys.path)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('iris.pdf')