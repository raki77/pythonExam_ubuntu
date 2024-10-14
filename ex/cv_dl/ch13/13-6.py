import numpy as np
from keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense,Reshape,Flatten,Conv2D,Conv2DTranspose,LeakyReLU,Dropout
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
x_train=x_train[np.isin(y_train,[8])]			# 부류 8(Bag) 추출
x_train=(x_train.astype('float32')/255.0)*2.0-1.0	# [-1,1] 정규화

zdim=100				# 잠복 공간의 차원

def make_discriminator(in_shape):	# 분별망 D
    model=Sequential()
    model.add(Conv2D(64,(3,3),padding='same',activation=LeakyReLU(alpha=0.2),input_shape=in_shape))
    model.add(Conv2D(128,(3,3),strides=(2,2),padding='same',activation=LeakyReLU(alpha=0.2)))
    model.add(Conv2D(128,(3,3),strides=(2,2),padding='same',activation=LeakyReLU(alpha=0.2)))
    model.add(Conv2D(256,(3,3),strides=(2,2),padding='same',activation=LeakyReLU(alpha=0.2)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002,beta_1=0.5),metrics=['accuracy'])
    return model

def make_generator(zdim):		# 생성망 G
    model=Sequential()
    model.add(Dense(7*7*64,activation=LeakyReLU(alpha=0.2),input_dim=zdim))
    model.add(Reshape((7,7,64)))
    model.add(Conv2D(128,(3,3),padding='same',activation=LeakyReLU(alpha=0.2)))
    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same',activation=LeakyReLU(alpha=0.2)))	# 14*14 업샘플링
    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same',activation=LeakyReLU(alpha=0.2)))	# 28*28 업샘플링
    model.add(Conv2D(1,(3,3),padding='same',activation='tanh'))
    return model

def make_gan(G,D):			# GAN 모델
    D.trainable=False
    model=Sequential()
    model.add(G)
    model.add(D)
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002,beta_1=0.5))
    return model

def generate_real_samples(dataset,n_samples):	# 진짜 샘플 뽑음
    ix=np.random.randint(0,dataset.shape[0],n_samples)
    x=dataset[ix]
    y=np.ones((n_samples,1))
    return x,y

def generate_latent_points(zdim,n_samples):	# 잠복 공간 점 생성
    return np.random.randn(n_samples,zdim)

def generate_fake_samples(G,zdim,n_samples):	# 가짜 샘플 생성
    x_input=generate_latent_points(zdim,n_samples)
    x=G.predict(x_input)
    y=np.zeros((n_samples,1))
    return x,y

def train(G,D,GAN,dataset,zdim,n_epochs=200,batch_siz=128,verbose=0): # GAN 학습
    n_batch=int(dataset.shape[0]/batch_siz)

    for epoch in range(n_epochs):
        for b in range(n_batch):
            x_real,y_real=generate_real_samples(dataset,batch_siz//2)
            d_loss1,_=D.train_on_batch(x_real,y_real)	# 진짜 샘플로 D 학습
            x_fake,y_fake=generate_fake_samples(G,zdim,batch_siz//2)
            d_loss2,_=D.train_on_batch(x_fake,y_fake)	# 가짜 샘플로 D 학습

            x_gan=generate_latent_points(zdim,batch_siz)
            y_gan=np.ones((batch_siz,1))
            g_loss=GAN.train_on_batch(x_gan,y_gan)	# G 학습
        if verbose==1:
            print('E%d: loss D(real)=%.3f, D(fake)%.3f GAN %.3f'%(epoch+1,d_loss1,d_loss2,g_loss))
        if (epoch+1)%10==0:
            x_fake,y_fake=generate_fake_samples(G,zdim,12)
            plt.figure(figsize=(20,2))
            plt.suptitle('epoch '+str(epoch+1))
            for k in range(12):
                plt.subplot(1,12,k+1)
                plt.imshow((x_fake[k]+1)/2.0,cmap='gray'); plt.xticks([]); plt.yticks([])
            plt.show()

D=make_discriminator((28,28,1))
G=make_generator(zdim)
GAN=make_gan(G,D)
train(G,D,GAN,x_train,zdim,verbose=1)