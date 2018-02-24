import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    
    tf_graph = tf.get_default_graph()
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # print names of elements in graph
    #for element in tf_graph.get_operations():
    #    print(element.name)
        
    # define names
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # load VGG pretrained graphs by name
    tf_input = tf_graph.get_tensor_by_name(vgg_input_tensor_name)
    tf_prob = tf_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    tf_layer3 = tf_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    tf_layer4 = tf_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    tf_layer7 = tf_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return tf_input, tf_prob, tf_layer3, tf_layer4, tf_layer7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    # create the network
    # use layers conv2d 1x1 kernel, conv2d_transpose, skip connections
    
    # encoder is vgg network

    # 1x1 convolutions for skip connections
    conv1x1_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides = (1,1), padding='same') 
    conv1x1_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides = (1,1), padding='same')
    conv1x1_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides = (1,1), padding='same')
    
    # conv2transpose deconvolutions to scale up and concatenation with skip connections
    # after each upscaling, do conv2d but without changing dimensions
    # upscaling is mirrored conv2d downsampling path (number of convolutions and resolutions)
    
    # first upscaling layer
    deconv_layer7 = tf.layers.conv2d_transpose(conv1x1_layer7, num_classes,  4, strides=(2, 2), padding='same')
    skip_layer7 = tf.add(deconv_layer7, conv1x1_layer4)
    
    # second upscaling layer
    deconv_layer4 = tf.layers.conv2d_transpose(skip_layer7, num_classes,  4, strides=(2, 2), padding='same')
    skip_layer4 = tf.add(deconv_layer4, conv1x1_layer3)
    
    output = tf.layers.conv2d_transpose(skip_layer4, num_classes,  16, strides=(8, 8), padding='same')

    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    # define optimizer for training
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    # run training for defined number of epochs with defined parameters
    for epoch in range(epochs):
        for curr_batch, (curr_image, curr_label) in enumerate(get_batches_fn(batch_size)):
            history, curr_loss = sess.run([train_op, cross_entropy_loss],
                                          feed_dict = {input_image: curr_image, 
                                                       correct_label: curr_label, 
                                                       keep_prob: 0.5, 
                                                       learning_rate: 1e-4})
            print("curr batch", curr_batch, "curr epoch ", epoch, "/", epochs, "curr loss ", curr_loss)
    
    pass

tests.test_train_nn(train_nn)


def run():
    # define number of classes
    num_classes = 2
    
    # define image shape
    #image_shape = (384, 1248)
    image_shape = (160, 576)
        
    # define training parameter
    epochs = 50
    batch_size = 2
    learning_rate = tf.placeholder(dtype = tf.float32) 
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])

    # define dataset paths
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        tf_input, tf_prob, tf_layer3, tf_layer4, tf_layer7 = load_vgg(sess, vgg_path)
        nn_last_layer = layers(tf_layer3, tf_layer4, tf_layer7, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        
        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, tf_input, correct_label, tf_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, tf_prob, tf_input)
        
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
