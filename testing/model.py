from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal, constant

import re

def relu(x): return Activation('relu')(x)

def conv(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def vgg_block(x, weight_decay):
    # Block 1
    x = conv(x, 64, 3, "conv1_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")

    # Block 3
    x = conv(x, 256, 3, "conv3_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1")

    # Block 4
    x = conv(x, 512, 3, "conv4_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2", (weight_decay, 0))
    x = relu(x)

    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)

    return x

def vgg_flow_block(x, weight_decay):
    # Block 1
    x = conv(x, 64, 3, "conv1_1f", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2f", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1f")

    # Block 2
    x = conv(x, 128, 3, "conv2_1f", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2f", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1f")

    # Block 3
    x = conv(x, 256, 3, "conv3_1f", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2f", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3f", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4f", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1f")

    # Block 4
    x = conv(x, 512, 3, "conv4_1f", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2f", (weight_decay, 0))
    x = relu(x)

    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPMf", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPMf", (weight_decay, 0))
    x = relu(x)

    return x


def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))

    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))

    return x


def apply_mask(x, mask1, mask2, num_p, stage, branch, np_branch1, np_branch2):
    w_name = "weight_stage%d_L%d" % (stage, branch)

    # TODO: we have branch number here why we made so strange check
    # assert np_branch1 != np_branch2 # we selecting branches by number of pafs, if they accidentally became the same it will be disaster

    if num_p == np_branch1:
        w = Multiply(name=w_name)([x, mask1])  # vec_weight
    elif num_p == np_branch2:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    else:
        assert False, "wrong number of layers num_p=%d " % num_p
    return w


def get_training_model(weight_decay, np_branch1, np_branch2, stages = 6, gpus = None):    
    img_input_shape = (None, None, 3)
    img_flow_input_shape = (None, None, 3)
    vec_flow_input_shape = (None, None, np_branch1)
    heat_input_shape = (None, None, np_branch2)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    img_flow_input = Input(shape=img_flow_input_shape)
    vec_flow_weight_input = Input(shape=vec_flow_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_flow_input)
    inputs.append(img_input)
    
    if np_branch1 > 0:
        inputs.append(vec_flow_weight_input)

    if np_branch2 > 0:
        inputs.append(heat_weight_input)

    #img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]
    img_normalized = img_input # will be done on augmentation stage
    img_flow_normalized = img_flow_input # will be done on augmentation stage

    # VGG
    vgg_merged = []

    stage0_out = vgg_block(img_normalized, weight_decay)
    stage0_flow_out = vgg_flow_block(img_flow_normalized, weight_decay)

    vgg_merged.append(stage0_out)
    vgg_merged.append(stage0_flow_out)

    vgg_merged_input = Concatenate()(vgg_merged)

    new_x = []

    # stage 1 - branch 1 (flow confidence maps)

    if np_branch1 > 0:
        stage1_branch1_out = stage1_block(vgg_merged_input, np_branch1, 1, weight_decay)
        w1 = apply_mask(stage1_branch1_out, vec_flow_weight_input, heat_weight_input, np_branch1, 1, 1, np_branch1, np_branch2)
        outputs.append(w1)
        new_x.append(stage1_branch1_out)

    # stage 1 - branch 2 (confidence maps)

    if np_branch2 > 0:
        stage1_branch2_out = stage1_block(vgg_merged_input, np_branch2, 2, weight_decay)
        w2 = apply_mask(stage1_branch2_out, vec_flow_weight_input, heat_weight_input, np_branch2, 1, 2, np_branch1, np_branch2)
        outputs.append(w2)
        new_x.append(stage1_branch2_out)

    new_x.append(vgg_merged_input)
    # new_x.append(stage0_flow_out)

    x = Concatenate()(new_x)

    # stage sn >= 2
    for sn in range(2, stages + 1):

        new_x = []
        # stage SN - branch 1 (flow confidence maps)
        if np_branch1 > 0:
            stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
            w1 = apply_mask(stageT_branch1_out, vec_flow_weight_input, heat_weight_input, np_branch1, sn, 1, np_branch1, np_branch2)
            outputs.append(w1)
            new_x.append(stageT_branch1_out)

        # stage SN - branch 2 (confidence maps)
        if np_branch2 > 0:
            stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
            w2 = apply_mask(stageT_branch2_out, vec_flow_weight_input, heat_weight_input, np_branch2, sn, 2, np_branch1, np_branch2)
            outputs.append(w2)
            new_x.append(stageT_branch2_out)

        new_x.append(vgg_merged_input)
        # new_x.append(stage0_out)

        if sn < stages:
            x = Concatenate()(new_x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_lrmult(model):

    # setup lr multipliers for conv layers
    lr_mult = dict()

    for layer in model.layers:

        if isinstance(layer, Conv2D):

            # stage = 1
            if re.match("Mconv\d_stage1.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

            # stage > 1
            elif re.match("Mconv\d_stage.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 4
                lr_mult[bias_name] = 8

            # vgg
            else:
               print("matched as vgg layer", layer.name)
               kernel_name = layer.weights[0].name
               bias_name = layer.weights[1].name
               lr_mult[kernel_name] = 1
               lr_mult[bias_name] = 2

    return lr_mult


def get_testing_model(np_branch1=52, np_branch2=27, stages = 6):

    img_input_shape = (None, None, 3)
    img_flow_input_shape = (None, None, 3)

    img_input = Input(shape=img_input_shape)
    img_flow_input = Input(shape=img_flow_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]
    img_flow_normalized = Lambda(lambda x: x / 256 - 0.5)(img_flow_input) # [-0.5, 0.5]

    # VGG
    vgg_merged = []

    stage0_out = vgg_block(img_normalized, None)
    stage0_flow_out = vgg_flow_block(img_flow_normalized, None)

    vgg_merged.append(stage0_out)
    vgg_merged.append(stage0_flow_out)

    vgg_merged_input = Concatenate()(vgg_merged)

    stages_out = []

    # stage 1 - branch 1 (PAF)
    if np_branch1 > 0:
        stage1_branch1_out = stage1_block(vgg_merged_input, np_branch1, 1, None)
        stages_out.append(stage1_branch1_out)

    # stage 1 - branch 2 (confidence maps)
    if np_branch2 > 0:
        stage1_branch2_out = stage1_block(vgg_merged_input, np_branch2, 2, None)
        stages_out.append(stage1_branch2_out)

    stages_out.append(vgg_merged_input)

    x = Concatenate()(stages_out)
    # x = Concatenate()(stage0_flow_out + [stage0_flow_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):

        stages_out = []

        if np_branch1 > 0:
            stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
            stages_out.append(stageT_branch1_out)
        if np_branch2 > 0:
            stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)
            stages_out.append(stageT_branch2_out)

        stages_out.append(vgg_merged_input)        

        if sn < stages:
            x = Concatenate()(stages_out)

    model = Model(inputs=[img_flow_input, img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model