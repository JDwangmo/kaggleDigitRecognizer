#encoding=utf8

import pandas as pd
import Image
import numpy as np
import  logging
import matplotlib.pyplot as plt
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )

def load_data(data_file_path,image_shape = (28,28),returnlabel = False):
    '''

    :param data_file_path:
    :param returnlabel: 是否返回标签
    :return:
    '''
    logging.debug('开始加载文件：%s'%(data_file_path))
    data = pd.read_csv(data_file_path,sep=',',header=0)

    X = data.filter(regex=r'pixel.*').as_matrix().astype(np.float)/255
    # print X[0]
    # quit()
    if len(image_shape)==1:
        logging.debug('图片shape为：%d,'%(image_shape))
    elif len(image_shape)==2:
        logging.debug('图片shape为：%d,%d'%(image_shape))
        X = X.reshape(len(data),image_shape[0],image_shape[1])
    else:
        logging.debug('转换图片shape有错！')

    # print test_pix[0]
    # print test_pix[0].shape
    # pic = Image.fromarray(test_pix[0])
    # pic.save('test.bmp','bmp')
    print data['label'].value_counts()
    if returnlabel:
        label = data['label']
        return X,label
    else:
        return X
if __name__=='__main__':
    test_file_path = '/home/jdwang/PycharmProjects/kaggleDigitRecognizer/train_test_data/' \
                     'test.csv'
    # test_X = load_data(test_file_path,
    #                       image_shape=(784, 1),
    #                       returnlabel=False)

    train_file_path = '/home/jdwang/PycharmProjects/kaggleDigitRecognizer/train_test_data/' \
                     'train.csv'
    train_X,train_y = load_data(train_file_path,
                                 image_shape=(784,1),
                                 returnlabel=True)
    # print train_X.shape
    # print train_X[0].reshape(28,28)
    # plt.imshow(train_X[0].reshape(28,28),cmap=plt.cm.binary)
    # plt.show()