import tensorflow as tf
import os
import numpy as np
import glob

import sys

tf.enable_eager_execution()

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = { # configuration for decoding tf_record file
            'dsize_dim0': tf.FixedLenFeature([], tf.int64),
            'dsize_dim1': tf.FixedLenFeature([], tf.int64),
            'dsize_dim2': tf.FixedLenFeature([], tf.int64),
            'lsize_dim0': tf.FixedLenFeature([], tf.int64),
            'lsize_dim1': tf.FixedLenFeature([], tf.int64),
            'lsize_dim2': tf.FixedLenFeature([], tf.int64),
            'data_vol': tf.FixedLenFeature([], tf.string),
            'label_vol': tf.FixedLenFeature([], tf.string)})
    return features

def convert_tf_record_numpy(input_folders, output_folder, output_prefix):
    """
    input_folders: input_folders containing data
    output_folders: images saved as numpy in subfolders images, labels
    """
    IMG_SIZE = [256, 256, 3]
    all_tf_files = []
    for folder in input_folders:
        all_tf_files += [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('tfrecords')]
    
    n_files = len(all_tf_files)
    
    if not os.path.exists(os.path.join(output_folder, 'images')):
        os.makedirs(os.path.join(output_folder, 'images'))
    
    if not os.path.exists(os.path.join(output_folder, 'labels')):
        os.makedirs(os.path.join(output_folder, 'labels'))
    
    
    
    file_queue = tf.train.string_input_producer(all_tf_files)
    
    with tf.Session() as sess:
        all_data = read_and_decode(file_queue)
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for i in range(n_files):
            sample = sess.run(all_data)
            img_vol = tf.decode_raw(sample['data_vol'], tf.float32)
            label_vol = tf.decode_raw(sample['label_vol'], tf.float32)
            
            img_vol = tf.reshape(img_vol, IMG_SIZE)
            label_vol = tf.reshape(label_vol, IMG_SIZE)
            
            # take a slice of size 3 for contextual information as mentioned in the paper
            img_vol = img_vol.eval()
            label_vol = label_vol.eval()
            
            label_vol = label_vol
            
            np.save(os.path.join(output_folder, 'images', output_prefix + str(i).zfill(6)), img_vol)
            np.save(os.path.join(output_folder, 'labels', output_prefix + str(i).zfill(6)), label_vol)
            
        coord.request_stop()
        coord.join(threads)

features = { # configuration for decoding tf_record file
            'dsize_dim0': tf.FixedLenFeature([], tf.int64),
            'dsize_dim1': tf.FixedLenFeature([], tf.int64),
            'dsize_dim2': tf.FixedLenFeature([], tf.int64),
            'lsize_dim0': tf.FixedLenFeature([], tf.int64),
            'lsize_dim1': tf.FixedLenFeature([], tf.int64),
            'lsize_dim2': tf.FixedLenFeature([], tf.int64),
            'data_vol': tf.FixedLenFeature([], tf.string),
            'label_vol': tf.FixedLenFeature([], tf.string)}

def _parse_image_function(example_proto):
 return tf.io.parse_single_example(example_proto, features)

def convert_tf_record_numpy_fast(input_folders, output_folder, output_prefix):
    if not os.path.exists(os.path.join(output_folder, 'images')):
        os.makedirs(os.path.join(output_folder, 'images'))
    
    if not os.path.exists(os.path.join(output_folder, 'labels')):
        os.makedirs(os.path.join(output_folder, 'labels'))
    
    IMG_SIZE = [256, 256, 3]
    all_tf_files = glob.glob(input_folders[0] + '/*.tfrecords')
    image_dataset = []
    print('reading tf_record files')
    for it, i in enumerate(all_tf_files):
        if it%1000 == 0:
            print('number of files read: ', it)
        image_dataset += [tf.data.TFRecordDataset(i).map(_parse_image_function)]

    for i, j in enumerate(image_dataset):
        for data in j:
            img_numpy = tf.decode_raw(data['data_vol'], tf.float32).numpy()
            label_numpy = tf.decode_raw(data['label_vol'], tf.float32).numpy()

            img_numpy = img_numpy.reshape(IMG_SIZE)
            label_numpy = label_numpy.reshape(IMG_SIZE)

            if np.sum(label_numpy) == 0:
                continue
            
            np.save(os.path.join(output_folder, 'images', output_prefix + str(i).zfill(6)), img_numpy)
            np.save(os.path.join(output_folder, 'labels', output_prefix + str(i).zfill(6)), label_numpy)

    

if __name__ == '__main__':
    modality = sys.argv[1]
    convert_tf_record_numpy(['./PnpAda_release_data/train&val/%s_train_tfs' % modality], modality+'_train', modality)
    convert_tf_record_numpy(['./PnpAda_release_data/train&val/%s_val_tfs' % modality], modality+'_test', modality)