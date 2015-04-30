
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
style.use('ggplot')

# model initilize
clf = svm.SVC(gamma=0.001, C=100)

# model fitting
clf.fit(digits.data[:-1], digits.target[:-1])
print('predict result: {predict_result}'.format(predict_result=clf.predict(digits.data[-1])))

# show the real img
img = digits.data[-1].reshape(8, 8)
plt.imshow(img)
# plt.show()

