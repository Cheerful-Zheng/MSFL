# import numpy as np
# from sklearn.metrics import mean_squared_error
# y_true = np.array([3, -0.5, 2, 7])
# y_pred = np.array([2.5, 0.0, 2, 8])
# print(mean_squared_error(y_true, y_pred))
# print(mean_squared_error(y_true, y_pred, squared=False))
# print(2**3)
#
#
# y = [3, -0.5, 2, 7]
# print(y[2])
# y.append([2, 3])
# print(y)
#
# from pyod.models.iforest import IForest   # imprt kNN分类器
#
# X_train = [[3, -0.5, 2, 7], [3, -0.5, 2, 7], [30, -0.5, 20, 7], [3, -0.5, 2, 7], [3, -0.5, 2, 7], [3, -0.5, 2, 7], [3, -0.5, 2, 7]]
# X_test = [[3, -0.5, 2, 7], [3, -0.5, 2, 7]]
#
# # 训练一个kNN检测器
# clf_name = 'kNN'
# clf = IForest()
# clf.fit(X_train) # 使用X_train训练检测器clf
#
# # 返回训练数据X_train上的异常标签和异常分值
# y_train_pred = clf.labels_  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
# y_train_scores = clf.decision_scores_  # 返回训练数据上的异常值 (分值越大越异常)
# print(y_train_pred)
# print(y_train_scores)
#
# # 用训练好的clf来预测未知数据中的异常值
# y_test_pred = clf.predict(X_test)  # 返回未知数据上的分类标签 (0: 正常值, 1: 异常值)
# y_test_scores = clf.decision_function(X_test)  #  返回未知数据上的异常值 (分值越大越异常)
# print(y_test_pred)
# print(y_test_scores)
# from numpy import mean

# x = [0, 0, 0, 1, 0]
# if 1 in x:
#     print(1)
# else:
#     print(0)
# print(mean([1,2]))

import numpy as np
import pandas as pd

# df_dict = {
#     'city':'北京,上海,广州,深圳,台北'.split(','),
#     'price':(68000,54000,35000,72000,50000),
#     'year':np.arange(2015,2020)
# }
# df = pd.DataFrame(df_dict)
#
# df2 = df['city']
#
# df['city'] = 'Shaanxi'
#
# print(df)

# print(list(range(20)))

# import tensorflow as tf
# import numpy as np
#
# # 创建变量w
# w = tf.Variable(np.array([[6.0, 8, 6]]), dtype=tf.float32, name='UserEmbedding/UserEmbeddingKernel')
# w.assign(np.array([[1.0, 1, 1]]))
# # 查看变量的shape,而不是值。
# print(w)

a = ([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4])
print(a[1:2])
