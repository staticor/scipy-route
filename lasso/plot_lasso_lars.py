
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

print('Computing regularization path using the LARS...')
alphas, _ coefs = linear_model.lars_path(X, y, method='lasso', )