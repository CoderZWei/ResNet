from tensorlayer.layers import *
g_init=tf.random_normal_initializer(1.,0.02)
b_init=None
w_init=tf.random_normal_initializer(stddev=0.02)
def _bn_relu_conv(input,filters,kernel_size,strides=[1,1],is_train=False,name=None):
    with tf.variable_scope(name):
        x=BatchNormLayer(input,act=tf.nn.relu,is_train=is_train,gamma_init=g_init)
        x=Conv2d(x,filters,(kernel_size,kernel_size),strides,act=None,padding='SAME',W_init=w_init,b_init=b_init)
    return x
def _conv_bn_relu(input,filters,kernel_size=3,strides=[1,1],is_train=False):
    with tf.variable_scope('conv_bn_relu'):
        x=Conv2d(input,filters,(kernel_size,kernel_size),strides,act=None,padding='SAME',W_init=w_init,b_init=b_init)
        x=BatchNormLayer(x,act=tf.nn.relu,is_train=is_train,gamma_init=g_init)
    return x
def basic_block(index,input,filters,strides=[1,1],is_first_block_of_first=False,is_train=False):
    with tf.variable_scope('basic_block_%s'%index):
        temp=Conv2d(input,filters,(1,1),strides,padding='SAME',name='first_conv_%s'%index)
        if is_first_block_of_first:
            x=Conv2d(input,filters,(3,3),strides,padding='SAME',name='aaa')
        else:
            x=_bn_relu_conv(input,filters,kernel_size=3,is_train=is_train,name='bbb')
        x=_bn_relu_conv(x,filters=filters,kernel_size=3,strides=strides,is_train=is_train,name='bn_relu_conv')
        x=ElementwiseLayer([temp,x],tf.add)
    return x


def bottleneck_block(input,filters,strides=[1,1],is_first_block_of_first=False,is_train=False):
    temp=tf.layers.conv2d(input,filters=filters,kernel_size=1,strides=strides,padding='SAME')
    if is_first_block_of_first:
        x=tf.layers.conv2d(input,filters=filters,kernel_size=1,strides=strides,padding='SAME')
    else:
        x=_bn_relu_conv(input,filters,kernel_size=1)
    x=_bn_relu_conv(x,filters,kernel_size=3)
    x=_bn_relu_conv(x,filters=filters*4,kernel_size=1)
    x=tf.add(temp,x)
    return x

def residual_block(index,input,filters,block_type,repetitions,is_first_block,is_train=False):
    with tf.variable_scope('residual_block_%s'%index):
        #temp=input
        if block_type=='basic':
            for i in range(repetitions):
                strides=[1,1]
                if i==0 and not is_first_block:
                    strides=[2,2]
                x=basic_block(i,input,filters,strides,is_first_block_of_first=(is_first_block and i==0),is_train=is_train)
            return x
        else:
            for i in range(repetitions):
                strides=[1,1]
                if i==0 and not is_first_block:
                    strides=[2,2]
                x=bottleneck_block(input,filters,strides,is_first_block_of_first=(is_first_block and i==0),is_train=is_train)
            return x

def build(input,num_layer,filters,repetitions,n_classes,is_train=False):
    x=InputLayer(input)
    x=_conv_bn_relu(x,filters=64,kernel_size=7,strides=[2,2],is_train=is_train)
    x=MaxPool2d(x,filter_size=(3,3),strides=(2,2),padding='SAME')
    #temp=x
    if num_layer<50:
        block_type='basic'
    else:
        block_type='bottleneck'
    for i,r in enumerate(repetitions):
        x=residual_block(i,x,filters,block_type=block_type,repetitions=r,is_first_block=(i==0),is_train=is_train)
        filters*=2
    x=BatchNormLayer(x,act=tf.nn.relu,is_train=is_train,gamma_init=g_init)
    shape1=x.outputs.shape
    x=MeanPool2d(x,filter_size=(shape1[1],shape1[2]),strides=(1,1))
    x=FlattenLayer(x)
    logits=DenseLayer(x,n_units=n_classes)
    output=tf.nn.softmax(logits.outputs)
    return logits.outputs,output



