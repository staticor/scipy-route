
# Lasso


The *Lasso* is a linear model that estimates sparse coefiicients.

It is useful in some contexts due to its tendency to prefer solutions with few parameter values, effectively reducing the number of variables upon which the given solution is dependent.


Mathematically, the objective function to minimize is:
![lasso-formula](http://scikit-learn.org/stable/_images/math/5ff15825a85204658e3e5aa6e3b5952b8f709c27.png)

Sort of like Ridge Regression, Lasso added alpha item to be as the least-squares penalty, where alpha is a constant and ||w_1|| is the l1_norm of the parameter vector.

 
```

from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
print(clf.fit([[0, 0], [1, 1]], [0, 1]))

'''output

Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute='auto', tol=0.0001,
   warm_start=False)

'''
```


## Lasso Model Frame in sklearn

> sklearn.linear_model.Lasso

### Initial

Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')


* alpha: float, optional, defaults to 1.0
    Constant that multiplies L1 term. `alpha=0` is equivalent to an OLS model. For numerical reasons, using `alpha=0` is with the Lasso object is not advised.
* fit_intercept: boolean
   wheter to calculate the intercept for this model, if true, data is expected to be all centred.

* normalize: boolean, optional, default False
    if True, X will be normalized before regression.
* copy_X: boolean, optional, default True
    if True, X will be copied; else, it may be overwritten.
* precopmute: {True, False, 'auto', array-like}
    wheter to use a precomputed Gram matrix to speed up calculations.
    If set to 'auto' let us decide.     
    warning: 'auto' option is deprecated and will be removed in 0.18.
* max_iter:int, optional  
    
* tol: float, optional
    tolerance
* warm_start: bool, optional
    if True, reuse the solution of the previous call to fit as initialization; otherwise, just erase and renew.
* positive: bool, optional
    Wehn set to True, forces the coefficients to be positive.
* selection: str, default 'cyclic'
    if set to 'random', a random coefficient is updated every iteration; 
    if set to 'cyclic', it turns slower, follow the rule of updating coefficient in order. 

* random_state: int, RandomState instance, or None (default)



### Attributes

* coef_ : parameter vector, w in the cost function formula
* sparse_coef_:   a readonly property derived from coef_
* intercept_
* n_iter_  : number of iterations. 

### Functions

* decision_function(X): Decision function of the linear model. 
* fit(X, y): Fit model with coordinate descent.
* get_params([deep])
* path(X, y[...optional parameters])
* predict(X): Predict using the linear model.
* score(X, y)
* set_params(**params)


## Questions?

到这里我就有了问题,  在创建Lasso时 如何选择一个合适的alpha参数???
多大就好? 多小就不好?? \


### Using cross-validation
sklearn exposes objects that set the Lasso *alpha* parameter by cv.
**LassoCV** and **LassoLarsCV.LassoLarsCV** is based on the Least Angle Rregression algorithm explained below.




### Information-Criteria based model selection

Alternatively, the estimator **LassoLarsIC** proposed to use the **Akaike information criterion(AIC)** and the **Bayes Information criterion(BIC)**. It is a computationally cheaper alternative to fidn the optimal value to alpha as the regularization  path is coputed only once instead of k+1 times when using k-fold CV. 

However, such criteria needs a proper estimation of the degress of freedom of the solution, are derived for large samples. 

[Examples of this](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#example-linear-model-plot-lasso-model-selection-py)

## Example code

* Compressive sensing: tomography reconstruction with L1 prior (LASSO)
[page](http://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html)




* Cross-validataion on diabetes dataset exercise

[page](http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html)

[code]()
![](http://ww2.sinaimg.cn/bmiddle/5810d07bjw1erl461j715j2086066mxg.jpg)

* 


