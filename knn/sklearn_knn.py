#encoding=utf8

from sklearn.neighbors import KNeighborsClassifier
from preprocessing.load_data import load_data
import logging
import numpy as np
import pandas as pd
import timeit

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )
# 设置训练数据和测试数据的路径

test_file_path = '/home/jdwang/PycharmProjects/kaggleDigitRecognizer/train_test_data/' \
                 'test.csv'
test_X = load_data(test_file_path,
                   image_shape=(784, ),
                   returnlabel=False)

train_file_path = '/home/jdwang/PycharmProjects/kaggleDigitRecognizer/train_test_data/' \
                  'train.csv'
train_X, train_y = load_data(train_file_path,
                             image_shape=(784, ),
                             returnlabel=True)

logging.debug( 'the shape of train sample:%d,%d'%(train_X.shape))
logging.debug( 'the shape of test sample:%d,%d'%(test_X.shape))

rand_list = np.random.RandomState(0).permutation(len(train_X))
vc_split = 0.99
num_train = int(len(train_X)*vc_split)
dev_X = train_X[rand_list][:num_train]
dev_y = train_y[rand_list][:num_train]
val_X = train_X[rand_list][num_train:]
val_y = train_y[rand_list][num_train:]
logging.debug('随机选择%d个sample作为训练'%(num_train))
logging.debug('随机选择%d个sample作为验证'%(len(val_X)))
# print dev_X.shape
# print val_X.shape


start = timeit.default_timer()

model = KNeighborsClassifier(n_neighbors=5,
                             weights='distance',
                             algorithm='kd_tree',
                             leaf_size=30,
                             p=2,
                             metric='minkowski',
                             metric_params=None,
                             n_jobs=10
                             )

model.fit(dev_X,dev_y)
pred_result = model.predict(val_X)

is_correct = (pred_result==val_y)
print '正确的个数：%d'%(sum(is_correct))
print '正确率：%f'%(sum(is_correct)/(len(val_y)*1.0))

# 预测
pred_result = model.predict(test_X)
test_result = pd.DataFrame({
            'ImageId':range(1,len(pred_result)+1),
            'Label':pred_result,
            })

test_result_path = '/home/jdwang/PycharmProjects/kaggleDigitRecognizer/knn/result/20160504/' \
                   'sklearn_knn_result_%d.csv'%(num_train)
test_result.to_csv(test_result_path,sep=',',index=False)

end = timeit.default_timer()
logging.debug('总共运行时间:%ds' % (end-start))
