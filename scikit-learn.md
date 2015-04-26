Introduction of Sklearn
----


# scikit-learn

## install 

The [install page](http://scikit-learn.org/stable/install.html) provides varies of routes of install up. 
For me, it's simple, just typing this line on Terminal:
`pip install -U scikit-learn`

sklearn包的安装很简单， 参考本链接。 

* import 

There have many useful sub-libraries in scikit-learn(i.e. sklearn for short). As usual, **import \*** is absolutely not a good way to do that. Just import those libraries or specified functions to be used. 

from sklearn import xxx
 
如无特别注明， 建议用此形式对sklearn中的包进行引用。 

## dataset 


sklearn have prepared some datasets, for instance the [iris](http://en.wikipedia.org/wiki/Iris_flower_data_set) [digits](http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits) 



`Digits`
Pen-Based Recognition of Handwritten Digits Dataset
10992 instances, 16 Attributes, No missing values,
This is a simple testcase for the Sample dataset of some classifier modeling.

`Iris`
Another famous dataset made up of iris of three related species.  (setosa, virginica and versicolor)
50 samples from each of three species.
I think some students have used it for linear discriminant analysis.

`diabetes`
The diabetes dataset consists of 10 healthy variable related to deabetes (age, sex, weights, blood pressure and so on) to messuare more than 400 patients, and recording an indication of disease progression after one year as labelled target value.


```

# Diabetes dataset
# Prepare for modeling -- splitting dataset
diabetes = datasets.load_diabetes()
indices = np.random.permutation(len(diabetes.data))
diabetes_X_train = diabetes.data[indices[:-20]]
diabetes_X_test = diabetes.data[indices[-20:]]
diabetes_y_train = diabetes.target[indices[:-20]]
diabetes_y_test = diabetes.target[indices[-20:]]

```

sklearn已经内置了一些经典的数据集， 如用于判别分析的IRIS, 常用于SVM的digits等等。 每套数据集都可以用load_xxx来导入。 

### load_data

Build-in datasets are imported throuthis:
`from sklearn import datasets`

```
iris = datasets.load_iris()

digits = datasets.load_digits()
```
And then you could use these two datasets.

### datasets structure
Dataset is like a dictionary-like object that holds all meta-data. Here we go:

```
# show digits' data and shape
print(digits.data)
print(len(digits.data[0]), len(digits.data))
print(digits.target, len(digits.target))

```

一般来说是用.data, .target来区别数据集中的各元素。

### Simple Case

```
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(digits.data[:-1], digits.target[:-1])
print('predict result: {ele}'.format(ele=clf.predict(digits.data[-1])))
import matplotlib.pyplot as plt
img = digits.data[-1].reshape(8, 8)
plt.imshow(img)
plt.show()
```

This code gives a SVM classfier based the digits dataset. 
Training set is the slice [:-1];
Test set is only the last one. 

Using matplotlib to show the raw image. 


### picle

When you built a model, you could store it and use it for different situations by picle. 

[The python code is uploaded in github](https://github.com/staticor/scipy-route/tree/master/dataset).





## Machine Learning Introduction

下面是对机器学习的简单介绍， 很浅显， 希望能够对你有所帮助。 

What is the problem setting of machine learning?
> In general, a learning problem considers a set of samples of data and then tries to predict the unknown data's properties.

There has a few large categories:

* supervised learning
  Here is the [page](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning) of methods provided by scikit learn.
  Shortly to say, it is a kind of problem to predict.
  - classification: When we want to predict the number of unknown dataset's category, it is a classification problem. The target vector is always a finite number of discrete categories. Use case: to identify a client is good or bad for some credit-loan companies.
  - regression: In other cases, we want to predict a precise value, such as the price of house, the salary, and so on.   
* unsupervised learning

  If your input sets is all x without corresponding target values\label, and the goal in such problems maybe to find similar examples within the data, where it is called **clustering**, or to give prediction to the whole input, known as **density estimation**.
  
  
I think supervised learning is my first step to deep in.

## Training set and Testing set

Simply to understand, the training set is to build a model, being applied to the testing set.

对于数据集的划分， 对于supervised类的模型常把数据集分成训练集和预测集。 



### Exmaples of supervised learning

`predicting an output variable from high-dimensional observations`
Supervised learning is to find the link between two datasets: input X and y to predict (also called labels). Most often, y is a 1D array data. 


### The curse of dimensionality 

For an estimator to be effective, you need the distance between neighboring points to be less some value **d**, which depends on the problem. 
In one dimension, this requires on average *n/d*. In the KNN context, if data is described by just one feature with values ranging from 0 to 1, and with n training observations, then new data will be no further away from 1/n. Therefore, the NN decision rule will be efficient as soon as 1/n si small compared to the scale of between-class feature variations.

If the number of features is *p*, such as p=10, you now require d**10 points in 10 dimensions to pave the [0, 1] space. As p becomes large, the number of training points required for a good estimator.

这一部分的描述我自己还没有完全理会， 参考[wiki](http://en.wikipedia.org/wiki/Curse_of_dimensionality)的内容也许能有更多收获。 

## Linear Regression

`Linear model: regression`
![](http://ww3.sinaimg.cn/large/5810d07bjw1eriryh90u2j20hq0dyglw.jpg)
Use diabetes as example, to find the regression line in this case. 


```


from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
#LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
print(regr.coef_)

# The mean square error
print(np.mean( (regr.predict(diabetes_X_test) - diabetes_y_test )**2 ))
# MSE: 2413.65599655
# Explained Variance Score: 1 is the perfect prediction, and 0 is worst -> There is no linear relationship -> You choose wrong estimator.

print(regr.score(diabetes_X_test, diabetes_y_test))
# 0.43

x_indices = np.arange(len(diabetes_X_test))

import matplotlib.pyplot as plt
plt.scatter(x_indices, regr.predict(diabetes_X_test), color='g')
plt.scatter(x_indices, diabetes_y_test, color='r')
plt.show()

```
这段代码是对简单线性回归OLS模型的操作， 利用了diabetes数据， 数据集划分是随机排列后取后20作Test set. 
建模后regr封装了线性回归模型的系数， 模型评分（R Square) 等信息

### Shrinkage  -  Ridge Regression

If there are few data points per dimension, noise in the observations induces high variance:

```

import numpy as np
X = np.c_[.5, 1].T
y = [.5, 1]
test = np.c_[0, 2].T
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
import pylab as pl
pl.figure()

np.random.seed(1424)
for _ in range(6):
    this_X = 0.1 * np.random.normal(size=(2, 1)) + X
    regr.fit(this_X, y)
    pl.plot(test, regr.predict(test))
    pl.scatter(this_X, y, s=3)

pl.show()
```

In high-dimensional statistical learning is to shrink the regression coefficient to zero: any two randomly chosen set of observations are likely to be uncorrelated.

在数据量少而维度多时， 随机效应作会产生高方差的现象。 

用Ridge回归是一种可采取的应对策略：

```
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
import pylab as pl
pl.figure()
import numpy as np
X = np.c_[.5, 1].T
y = [.5, 1]
test = np.c_[0, 2].T
np.random.seed(1428)
for _ in range(6):
    this_X = 0.1 * np.random.normal(size=(2,1)) + X
    regr.fit(this_X, y)
    pl.plot(test, regr.predict(test))
    pl.scatter(this_X, y, s=3)

pl.show()

# trade off
regr = linear_model.Ridge(alpha=0.1)
pl.figure()
np.random.seed(0)
for _ in range(6):
    this_X = 0.1 * np.random.normal(size=(2,1)) + X
    regr.fit(this_X, y)
    pl.plot(test, regr.predict(test))
    pl.scatter(this_X, y, s=3)
```

Ridge回归中的参数alpha， 实际上是对模型无偏性造成影响， alpha值越大， 模型越有偏， 方差越小。 

### Lasso
Lasso is the abbreviation of **Least Absolute Shrinkage and Selection Operator**, can set some coefficients to zero.
Such methods are called sparse method and sparsity can be seen as an application of Occam's razor : prefer simpler models.



```

from __future__ import print_function

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import pylab as pl
pl.figure()
import numpy as np
from sklearn import datasets
diabetes = datasets.load_diabetes()
np.random.seed(1424)
indices = np.random.permutation(len(diabetes.data))
diabetes_X_train = diabetes.data[indices[:-20]]
diabetes_X_test = diabetes.data[indices[-20:]]
diabetes_y_train = diabetes.target[indices[:-20]]
diabetes_y_test = diabetes.target[indices[-20:]]

alphas = np.logspace(-4, -1, 6)
regr = linear_model.Lasso()
scores = [regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train).score(diabetes_X_test, diabetes_y_test) for alpha in alphas]

best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha

print(regr.fit(diabetes_X_train, diabetes_y_train))
print(regr.coef_)
```

`Different algorithms for the same problem`
For the same mathematical problem, we can use different algorithms. For instance the *Lasso* object in scikit-learn solves the *lasso* regression problem using a coordinate decent method, that is efficient on large datasets. 
However, there is another algorithm named *LassoLars* using the **LARS**, which is very efficient for problems in which the weight vector estimated is ver sparse (i.e. problems with very few observations).




## Classification

Here we will use *iris* datasets, in which linear regression is not the right approach. 

But someone have used linear regression's function to be as a classifier -- a sigmoid function or logistic function like follows:  

![img](http://ww3.sinaimg.cn/bmiddle/5810d07bjw1erj2bx1kg3j20cu015746.jpg)

`Logistic Regression`

```
logistic = linear_model.LogisticRegression(C=1e5)
print(logistic.fit(iris_X_train, iris_y_train))

```

Often to say, the logistic regression is to be used as a classifier of binary classification problem.

#### Logistic Regression in Sklearn

[Sklearn Doc](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
`sklearn.linear_model.LogisticRegression`

`model Parameter`

* penalty=   'l1'  or  'l2'
		Used to specify the norm used in the penalization.  (Newton-Cg and lbfgs solvers support only l2 penalties)
* dual=   True or False
		Dual or Primal formulation.  Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
* tol=  FLOAT optional
		tolerance for stopping criteria.  「迭代停止准则」
* C= 1.0 (default)  positive float
		Inverse of regularization strength. Smaller values specify strnger regularization. 
* fit_intercept= True(default) or False
		if model is with bias(y-intercept is not equal to zero), True. 
* intercept_scaling = 1 (default)  |float
		useful only if solver is liblinear. 
* class_weight= {dict, 'auto'}  
		The 'auto' mode selects weights inversely proportional to class frequencies in the training set. 
* random_state=  INT seed,  RandomState instance, or None(default)
		The seed of pseudo random number generator to use when shuffling the data. 
* solver= {'newton-cg', 'lbfgs', 'liblinear'}
		Algorithm to use in the optimization problem. 
* max_iter=  INT type
		useful only for the newton-cg and lbfgs solvers. Maximum number of iterations taken for the solvers to converge. 
* multi_class= {'ovr', 'multinomial'}
		ovr: a binary problem is fit for each label. 
		else (multi class), the loss minimised is the multinomial loss fit across the entire probability distribution. Works only for the 'lbfgs' solver.
* verbose= INT
		For the liblinear and lbfgs solvers set verbose to any positive number fo verbosity.
		
		
== == == == == == == == == == == == == == ==

modeler attributes:

* coef_: like ols model, type-array | shape (n_classes, n_features). 
* intercept_ : array, shape(n_classes). 
* n_iter_ int,   Maximum of the actual number of iteration.



	

## Support Vector Machines(SVMs) 

SVMs belongs to the discriminant model family. They try to find a line combining samples to build a plane maximizing the margin between two classes.

`Regularization`
Regularization is set by the *c* parameter: smaller *c* means the margins is calculated using many or all observations around the separating line; larger value *c* means the margin is calculated on observations close to the line (less regularization). 

![](http://ww2.sinaimg.cn/large/5810d07bjw1erjayrinmdj215q0fuju3.jpg)


### Example: [Plot-Iris](http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#example-svm-plot-iris-py)


At first We consider the 2 features of this dataset:
* Separ length
* Separ width

We'll show how to plot the decision surface(hyper-line) for four SVM classifiers with different kernels.

`LinearSVC`

The Linear Models **LinearSVC()** and **SVC(kernel='linear')** yield slight different decision boundaries. 

* LinearSVC minimizes the squared hinge loss while minimize the regular hinge loss;
* LinearSVC uses the One-vs-All (also known as One-vs-Rest) multiclass reduction while SVC uses the One-vs-One multiclass reduction. 

`Linear vs Non-linear model`

Linear Model often gives linear boundaries when being classifier. Whereas, non-linear model are more flexible with shapes that depend on kind of kernel and its parameters, such polynomial or Gaussian RBF model. 

code: [plot_iris.py](https://raw.githubusercontent.com/staticor/scipy-route/bf0ddd833b5eb43a1d09b370eb53c6f79580d97c/dataset/plot_iris.py)



![](http://ww2.sinaimg.cn/large/5810d07bjw1erjfd8e2mkj20v20oyn29.jpg)




### SVR
SVMs are often used as typical classifier, however, they actually also can be used in regression. 
So in sklearn you will import svm, and then use its svc function. 

* from sklearn import svm
* svc = svm.SVC(...)


`Normalizing Data`
For many situations, including SVMs, having datasets with unit std. for each feature is important to get good prediction.

`SVM's kernel`

* svm.SVC(kernel='linear')
* ... kernel = 'poly', degree=3
* ... kernel = 'rbf'   # Radial Basis Function


#### Libsvm 

`GUI`

[libsvm_gui](http://scikit-learn.org/stable/auto_examples/applications/svm_gui.html#example-applications-svm-gui-py)

## Model Selection

`How to choose the right\better estimator`

Often the toughest task of solving a machine problem can be finding the estimator.

A Chinese saying goes "因地制宜"， which means methods  
are of different effects for different types of data and problems.
In this picture you can get a first eye of use guiding.
![img](http://ww1.sinaimg.cn/bmiddle/5810d07bjw1erhsl53jdqj21kw0xzws3.jpg)



关于模型选择, [sklearn](http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html) 给出了相关的建议。

