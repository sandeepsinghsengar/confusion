
--------------
"""
from argparse import ArgumentParser
import os
import tensorflow as tf
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#tf.config.set_visible_devices([], 'GPU')
#import tensorflow.compat.v1 as tf
import numpy as np
import time
from tqdm import tqdm
import shutil
import logging
import argparse
from importlib.machinery import SourceFileLoader
from mpunet.preprocessing.data_loader import get_train_generators
from mpunet.models.probabilistic_unet import ProbUNet
import mpunet.utils.training_utils as training_utils
#from mpunet.utils.training_utils import cal_metrics
from mpunet.callbacks import init_callback_objects
from mpunet.logging import ScreenLogger
from mpunet.callbacks import (SavePredictionImages, Validation,
                              FGBatchBalancer, DividerLine,
                              LearningCurve, MemoryConsumption,
                              MeanReduceLogArrays, remove_validation_callbacks)
tf.compat.v1.disable_eager_execution() #tf placeholder cannot run in eager execusion enabled mode.To execute the operation/function, you always need to use like with tf.Session() as sess: print(sess.run(q, feed_dict={num: 10}). details at https://stackoverflow.com/questions/56561734/runtimeerror-tf-placeholder-is-not-compatible-with-eager-execution. Statements using tf.placeholder are no longer valid for tf 2.
def train1(cf):
    os.environ["CUDA_VISIBLE_DEVICES"] = cf.cuda_visible_devices

    # initialize data providers
    data_provider = get_train_generators(cf)
    train_provider = data_provider['train']
    val_provider = data_provider['val']

    prob_unet = ProbUNet(latent_dim=cf.latent_dim, num_channels=cf.num_channels,
                         num_1x1_convs=cf.num_1x1_convs,
                         num_classes=cf.num_classes, num_convs_per_block=cf.num_convs_per_block,
                         initializers={'w': training_utils.he_normal(),
                                       'b': tf.compat.v1.truncated_normal_initializer(stddev=0.001)},
                         regularizers={'w': tf.keras.regularizers.l2(0.5 * (1.0))})

    x = tf.compat.v1.placeholder(tf.float32, shape=cf.network_input_shape) #input image
    #print(x)
    y = tf.compat.v1.placeholder(tf.uint8, shape=cf.label_shape) #actual output
    #print(y)
    mask = tf.compat.v1.placeholder(tf.uint8, shape=cf.loss_mask_shape)
    #print(mask)
    global_step = tf.compat.v1.train.get_or_create_global_step()

    if cf.learning_rate_schedule == 'piecewise_constant':
        learning_rate = tf.compat.v1.train.piecewise_constant(x=global_step, **cf.learning_rate_kwargs)
    else:
        learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=cf.initial_learning_rate, global_step=global_step,
                                                   **cf.learning_rate_kwargs)
    with tf.device(cf.gpu_device):
        prob_unet(x, y, is_training=True, one_hot_labels=cf.one_hot_labels)
        prob_unet._build(x, y, is_training=True, one_hot_labels=cf.one_hot_labels)
        elbo = prob_unet.elbo(y, reconstruct_posterior_mean=cf.use_posterior_mean, beta=cf.beta, loss_mask=mask,
                              analytic_kl=cf.analytic_kl, one_hot_labels=cf.one_hot_labels)
        reconstructed_logits = prob_unet._rec_logits # predicted output, we can say y cap
        #confusion_matrix=prob_unet._conf_matrix
        #print(confusion_matrix[0,0])
        #performance_metrics=metrics(confusion_matrix)
        sampled_logits = prob_unet.sample()
        reg_loss = cf.regularizarion_weight * tf.reduce_sum(input_tensor=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        #print('get collection=',tf.reduce_sum(input_tensor=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)))
        loss = -elbo + reg_loss
        rec_loss = prob_unet._rec_loss_mean #reconstructed loss
        accuracy =prob_unet._accuracy_mean
        #print('accuracy and type =',accuracy, type(accuracy))
        precision=prob_unet._precision #all class precision values are in the form of dictionary
        print('precision and type =',precision, type(precision))
        print('precision 0=',precision[0])
        kl = prob_unet._kl
        mean_val_accuracy = tf.compat.v1.placeholder(tf.float32, shape=(), name="mean_val_accuracy")
        mean_val_rec_loss = tf.compat.v1.placeholder(tf.float32, shape=(), name="mean_val_rec_loss")
        mean_val_kl = tf.compat.v1.placeholder(tf.float32, shape=(), name="mean_val_kl")
        mean_val_precision=[]
        for i in range(cf.num_classes):
        	mean_val_precision.append(tf.compat.v1.placeholder(tf.float32, shape=(), name="mean_val_precision"))
        #mean_val_recall = tf.compat.v1.placeholder(tf.float32, shape=(), name="mean_val_recall")
        #mean_val_specificity = tf.compat.v1.placeholder(tf.float32, shape=(), name="mean_val_specificity")
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step) #adam optimizer is used to minimize the 'loss' value computed from elbo and reg_loss

    # prepare tf summaries
    train_elbo_summary = tf.compat.v1.summary.scalar('train_elbo', elbo) #summary is just used to log the data, summary.scalar return the single value
    train_kl_summary = tf.compat.v1.summary.scalar('train_kl', kl)
    train_rec_loss_summary = tf.compat.v1.summary.scalar('rec_loss', rec_loss)
    train_accuracy_summary = tf.compat.v1.summary.scalar('accuracy', accuracy)
    train_precision_summary=[]
    #for i in range(cf.num_classes):
    #	train_precision_summary.append(tf.compat.v1.summary.scalar('precision', precision [i]))

    #train_recall_summary = tf.compat.v1.summary.scalar('recall', confusion_matrix['recall'])
    #train_specificity_summary = tf.compat.v1.summary.scalar('specificity', confusion_matrix['specificity'])
    train_loss_summary = tf.compat.v1.summary.scalar('train_loss', loss)
    reg_loss_summary = tf.compat.v1.summary.scalar('train_reg_loss', reg_loss)
    lr_summary = tf.compat.v1.summary.scalar('learning_rate', learning_rate)
    beta_summary = tf.compat.v1.summary.scalar('beta', cf.beta)
    training_summary_op = tf.compat.v1.summary.merge([train_loss_summary, reg_loss_summary, lr_summary, train_elbo_summary,
    train_kl_summary, train_rec_loss_summary, train_accuracy_summary, beta_summary])
    batches_per_second = tf.compat.v1.placeholder(tf.float32, shape=(), name="batches_per_sec_placeholder")
    timing_summary = tf.compat.v1.summary.scalar('batches_per_sec', batches_per_second)
    val_rec_loss_summary = tf.compat.v1.summary.scalar('val_loss', mean_val_rec_loss)
    val_accuracy_summary = tf.compat.v1.summary.scalar('accuracy', mean_val_accuracy)
    #val_precision_summary=[]
    #for i in range(cf.num_classes):
    #	val_precision_summary.append(tf.compat.v1.summary.scalar('precision', mean_val_precision[i]))
    #val_recall_summary = tf.compat.v1.summary.scalar('recall', mean_val_recall)
    #val_specificity_summary = tf.compat.v1.summary.scalar('specificity', mean_val_specificity)
    val_kl_summary = tf.compat.v1.summary.scalar('val_kl', mean_val_kl)
    validation_summary_op = tf.compat.v1.summary.merge([val_rec_loss_summary, val_accuracy_summary, val_kl_summary])

    #tf.compat.v1.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver_hook = tf.estimator.CheckpointSaverHook(checkpoint_dir=cf.exp_dir, save_steps=cf.save_every_n_steps,
                                              saver=tf.compat.v1.train.Saver(save_relative_paths=True))
    # save config
    shutil.copyfile(cf.config_path, os.path.join(cf.exp_dir, 'used_config.py'))
    init = tf.compat.v1.global_variables_initializer()	
    with tf.compat.v1.train.MonitoredTrainingSession(hooks=[saver_hook]) as sess:
        summary_writer = tf.compat.v1.summary.FileWriter(cf.exp_dir, sess.graph)
        logging.info('Model: {}'.format(cf.exp_dir))
        
        for i in tqdm(range(cf.n_training_batches), disable=cf.disable_progress_bar): 
            '''
            here n_training_batches means #epochs. Target of this loop is to run train and validation for selected number of batches. This selection will            depend on how many training and validation batches are assigned in every_n_batches and n_batches respectively. In addition to all, #images               are already given in batch_size
            '''
            start_time = time.time()
            train_batch = next(train_provider)
            _, train_summary = sess.run([optimizer, training_summary_op],
                                        feed_dict={x: train_batch['data'], y: train_batch['seg'],
                                                   mask: train_batch['loss_mask']}) #loss mask is representing the segmented image in 1s and 0s 
            summary_writer.add_summary(train_summary, i)
            time_delta = time.time() - start_time
            train_speed = sess.run(timing_summary, feed_dict={batches_per_second: 1. / time_delta})
            summary_writer.add_summary(train_speed, i)

            # validation
            if i % cf.validation['every_n_batches'] == 0:

                train_rec = sess.run(reconstructed_logits, feed_dict={x: train_batch['data'], y: train_batch['seg']})
                image_path = os.path.join(cf.exp_dir,
                                          'batch_{}_train_reconstructions.png'.format(i // cf.validation['every_n_batches']))
                training_utils.plot_batch(train_batch, train_rec, num_classes=cf.num_classes,
                                          cmap=cf.color_map, out_dir=image_path)

                running_mean_val_rec_loss = 0.
                running_mean_val_accuracy = 0.
                running_mean_val_kl = 0.
                def zerolistmaker(n):
                	listofzeros=[0]*n
                	return listofzeros
                running_mean_val_precision=zerolistmaker(cf.num_classes)
                print('running_mean_val_precision=',running_mean_val_precision)
                for j in range(cf.validation['n_batches']):
                    val_batch = next(val_provider)
                    val_rec, val_sample, val_rec_loss, val_accuracy, val_precision, val_kl =\
                        sess.run([reconstructed_logits, sampled_logits, rec_loss, accuracy, precision, kl],
                                  feed_dict={x: val_batch['data'], y: val_batch['seg'], mask: val_batch['loss_mask']})
                    #print('val_precision=',val_precision)
                    running_mean_val_rec_loss += val_rec_loss / cf.validation['n_batches']
                    running_mean_val_accuracy += val_accuracy / cf.validation['n_batches']
                    running_mean_val_kl += val_kl / cf.validation['n_batches']
                    #for tt in range(cf.num_classes):
                    #	running_mean_val_precision[tt]+= val_precision[tt]/cf.validation['n_batches']
                    if j == 0:
                        image_path = os.path.join(cf.exp_dir,
                                                  'batch_{}_val_reconstructions.png'.format(i // cf.validation['every_n_batches']))
                        training_utils.plot_batch(val_batch, val_rec, num_classes=cf.num_classes,
                                                  cmap=cf.color_map, out_dir=image_path)
                        image_path = os.path.join(cf.exp_dir,
                                                  'batch_{}_val_samples.png'.format(i // cf.validation['every_n_batches']))

                        for _ in range(3):
                            val_sample_ = sess.run(sampled_logits, feed_dict={x: val_batch['data'], y: val_batch['seg']})
                            val_sample = np.concatenate([val_sample, val_sample_], axis=1)

                        training_utils.plot_batch(val_batch, val_sample, num_classes=cf.num_classes,
                                                  cmap=cf.color_map, out_dir=image_path)
                val_summary = sess.run(validation_summary_op, feed_dict={mean_val_rec_loss: running_mean_val_rec_loss, mean_val_accuracy:                                                         running_mean_val_accuracy, mean_val_kl: running_mean_val_kl})
                #print('running_mean_val_rec_loss is=',running_mean_val_rec_loss)
                summary_writer.add_summary(val_summary, i)
                #if cf.disable_progress_bar:
                logging.info('Evaluating epoch {}/{}: validation loss={}, validation accuracy={}, kl={}'\
                                 .format(i, cf.n_training_batches, running_mean_val_rec_loss, running_mean_val_accuracy, running_mean_val_kl))

            #sess.run(init)    
            sess.run(global_step)
