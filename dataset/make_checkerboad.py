
from sklearn.datasets import make_checkerboard

data, rows, columns = make_checkerboard(shape=(300, 300), n_clusters=(4,3), noise =10, shuffle=False, random_state=0)

print(data.shape)
print(rows.shape)
