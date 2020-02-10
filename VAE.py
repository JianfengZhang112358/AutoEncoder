#coding = utf-8
'''
*Author     :Jianfeng Zhang
*e-mail     :13052931019@163.com
*Blog       :https://me.csdn.net/qq_39004111
*Github     :https://github.com/JianfengZhang112358
*Data       :2020.02.01
*Description:Variation AutoEncoder
'''

import keras.backend as K
from keras.layers import Input, Dense,Lambda,BatchNormalization
from keras.models import Model,Sequential,load_model
from keras.datasets import mnist
from keras import metrics
from keras.utils import np_utils
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import time
#忽略系统警告
warnings.filterwarnings('ignore')

#网络参数
original_dim = 784
latent_dim1 = 1024
latent_dim2 = 256
latent_dim3 = 64
feature_dim = 2
batch_size = 128
epochs = 6
noise_factor = 0.3

# np.set_printoptions(threshold=np.inf)
#加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, original_dim)
x_test  = x_test.reshape(-1, original_dim)
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32') / 255.
#加入高斯噪声
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)#输出限幅
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#================================搭建VAE网络结构=============================
#encoder layer
input_data = Input(shape=(original_dim,),name='encoder_input')
Encoder1 = Dense(latent_dim1,activation='relu',name='feature1')(input_data)
Encoder1 = BatchNormalization(name='BN1')(Encoder1)
Encoder2 = Dense(latent_dim2,activation='relu',name='feature2')(Encoder1)
Encoder2 = BatchNormalization(name='BN2')(Encoder2)
Encoder3 = Dense(latent_dim3,activation='relu',name='feature3')(Encoder2)
Z_mean  = Dense(feature_dim,name='Z_mean')(Encoder3)#???为什么不要加上激活函数
Z_ln_var= Dense(feature_dim,name='Z_ln_var')(Encoder3)

#sample from distribution of P(Z|X)
def sample(theta):
    z_mean,z_ln_var = theta
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    e = K.random_normal(shape=(batch, dim))#在标准正态分布中采样形成形状跟均值、方差一样的e后续得到Z
    Z = z_mean + e*K.exp(0.5*z_ln_var)#Z为服从特征隐藏正态分布采样形成的特征
    return Z

#由特征分布参数得到采样出来的高斯特征
Feature = Lambda(sample,output_shape=(feature_dim,),name='Latent_feature')([Z_mean,Z_ln_var])

#decoder layer
Decoder1 = Dense(latent_dim3,activation='relu',name='feature4')(Feature)
Decoder1 = BatchNormalization(name='BN3')(Decoder1)
Decoder2 = Dense(latent_dim2,activation='relu',name='feature5')(Decoder1)
Decoder2 = BatchNormalization(name='BN4')(Decoder2)
Decoder3 = Dense(latent_dim1,activation='relu',name='feature6')(Decoder2)
X_hat    = Dense(original_dim,activation='sigmoid',name='Out_data')(Decoder3)

VAE = Model(input_data,X_hat)#VAE总体模型
VAE.summary()
plot_model(VAE,to_file='E:\工业大数据\基础代码\AutoEncoder\VAE\VAE.png',show_shapes=True,show_layer_names=True)#画出模型的结构图

Encoder = Model(input_data,Feature)#编码器部分模型
Encoder.summary()
plot_model(Encoder,to_file='E:\工业大数据\基础代码\AutoEncoder\VAE\Encoder.png',show_shapes=True,show_layer_names=True)#画出模型的结构图

Mean = Model(input_data,Z_mean)
Var = Model(input_data,Z_ln_var)


#解码器部分模型
decoder_input = Input(shape=(feature_dim,),name='decoder_input')
decoder1 =  Dense(latent_dim2,activation='relu',name='feature3')(decoder_input)
decoder1 = BatchNormalization(name='BN2')(decoder1)
decoder2 = Dense(latent_dim1,activation='relu',name='feature4')(decoder1)
decoder_out= Dense(original_dim,activation='sigmoid',name='Out_data')(decoder2)
decoder = Model(decoder_input,decoder_out)
decoder.summary()
plot_model(decoder,to_file='E:\工业大数据\基础代码\AutoEncoder\VAE\Decoder.png',show_shapes=True,show_layer_names=True)#画出模型的结构图

#====================================编译并训练模型=======================================
#定义VAE损失函数
def myloss(y_pre,y_real):
    ordinary_loss = original_dim * metrics.mean_absolute_error(input_data, X_hat)#普通自编码器的均方误差，用来衡量输出与输入的差异
    KL_loss = - 0.5 * K.sum(1 + Z_ln_var - K.square(Z_mean) - K.exp(Z_ln_var), axis=-1)#KL散度损失，用来衡量输出数据的分布与已知数据先验分布的差异
    VAE_loss = K.mean(ordinary_loss + KL_loss)
    return VAE_loss
# VAE.add_loss(VAE_loss)
VAE.compile(optimizer='adam',loss=myloss)
s = time.time()
history = VAE.fit(x_train_noisy,x_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.3,
                  shuffle=True,verbose=1,
                  callbacks=[TensorBoard(log_dir='VAE_log')])
e = time.time()
print('Took Time: %.3f s'%(e-s))
VAE.save('VAE.h5')

# VAE = load_model('VAE.h5', custom_objects={'my_loss':my_loss})#model = load_model('1.h5', custom_objects={'my_loss':my_loss,'NestedLSTM': NestedLSTM})

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
# print(history.history['val_loss'])
# print(history.history['val_loss'][-1])

#两维特征层可视化
x_test_encoder = Encoder.predict(x_train_noisy)
mean_test = Mean.predict(x_test_noisy)
var_test = Var.predict(x_test_noisy)
plt.figure()
plt.scatter(x_test_encoder[:,0],x_test_encoder[:,1],c=y_train,s=1)
plt.colorbar()
# ax = plt.axes(projection='3d')
# a = ax.scatter(x_test_encoder[:,0],x_test_encoder[:,1],x_test_encoder[:,2],c=y_test)
# plt.colorbar(a)
plt.show()

print('feature:',x_test_encoder.shape)
print(np.std(x_test_encoder,axis=0,ddof=0))
print('==========================')
print('mean:',mean_test.shape)
print(mean_test)
print('==========================')
print('var:',var_test.shape)
print(var_test)


#==========================================绘制预测图=====================================
#画10张图
predict_img = VAE.predict(x_test_noisy)
n = 10  # how many digits we will display
plt.figure(figsize=(10, 2))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)#两行n列位置在第i+1个的子图
    plt.imshow(x_test_noisy[i+100].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(predict_img[i+100].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




