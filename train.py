"""
mpunet train script
Optimizes a mpunet model in a specified project folder

Typical usage:
--------------
mp init_project --name my_project --data_dir ~/path/to/data
cd my_project
mp train --num_GPUs=1
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


def get_argparser():
    parser = ArgumentParser(description='Fit a mpunet model defined '
                                        'in a project folder. '
                                        'Invoke "init_project" to start a '
                                        'new project.')
    parser.add_argument("--project_dir", type=str, default='./',
                        help="Path to a mpunet project directory. "
                             "Defaults to the current directory.")
    parser.add_argument("--num_GPUs", type=int, default=1, #change default=1, if running on slurm cluster
                        help="Number of GPUs to use for this job (default=1)")
    parser.add_argument("--force_GPU", type=str, default="",
                        help="Manually set the CUDA_VISIBLE_DEVICES env "
                             "variable to this value "
                             "(force a specific set of GPUs to be used)")
    parser.add_argument("--continue_training", action="store_true",
                        help="Continue the last training session")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite previous training session at the '
                             'project path')
    parser.add_argument("--just_one", action="store_true",
                        help="For testing purposes, run only on the first "
                             "training and validation samples.")
    parser.add_argument("--no_val", action="store_true",
                        help="Do not perform validation (must be set if no "
                             "validation set exists)")
    parser.add_argument("--no_images", action="store_true",
                        help="Do not save sample images during training")
    parser.add_argument("--debug", action="store_true",
                        help="Set tfbg CLI wrapper on the session object")
    parser.add_argument("--wait_for", type=str, default="",
                        help="Wait for PID to terminate before starting the "
                             "training process.")
    parser.add_argument("--train_images_per_epoch", type=int, default=2500,
                        help="Number of training images to sample in each "
                             "epoch")
    parser.add_argument("--val_images_per_epoch", type=int, default=3500,
                        help="Number of training images to sample in each "
                             "epoch")
    parser.add_argument('-cf','--config', type=str, default='prob_unet_config.py',
                        help='name of the python script defining the training configuration')
    parser.add_argument('-d','--data_dir',type=str, default='', 
    help="full path to the data, if empty the config's data_dir attribute is used")
    parser.add_argument("--max_loaded_images", type=int, default=None,
                        help="Set a maximum number of (training) images to "
                             "keep loaded in memory at a given time. Images "
                             "will be cycled every '--num_access slices.'. "
                             "Default=None (all loaded).")
    parser.add_argument("--num_access", type=int, default=50,
                        help="Only effective with --max_loaded_images set. "
                             "Sets the number of times an images stored in "
                             "memory may be accessed (e.g. for sampling an "
                             "image slice) before it is replaced by another "
                             "image. Higher values makes the data loader "
                             "less likely to block. Lower values ensures that "
                             "images are sampled across all images of the "
                             "dataset. Default=50.")
    return parser


def validate_project_dir(project_dir):
    if not os.path.exists(project_dir) or \
            not os.path.exists(os.path.join(project_dir, "train_hparams.yaml")):
        raise RuntimeError("The script was launched from directory:\n'%s'"
                           "\n... but this is not a valid project folder.\n\n"
                           "* Make sure to launch the script from within a "
                           "MultiPlanarNet project directory\n"
                           "* Make sure that the directory contains a "
                           "'train_hparams.yaml' file." % project_dir)


def validate_args(args):
    """
    Checks that the passed commandline arguments are valid

    Args:
        args: argparse arguments
    """
    if args.continue_training and args.overwrite:
        raise ValueError("Cannot both continue training and overwrite the "
                         "previous training session. Remove the --overwrite "
                         "flag if trying to continue a previous training "
                         "session.")
    if args.train_images_per_epoch <= 0:
        raise ValueError("train_images_per_epoch must be a positive integer")
    if args.val_images_per_epoch <= 0:
        raise ValueError("val_images_per_epoch must be a positive integer. "
                         "Use --no_val instead.")
    if args.force_GPU and args.num_GPUs != 1:
        raise ValueError("Should not specify both --force_GPU and --num_GPUs")
    if args.num_GPUs < 0:
        raise ValueError("num_GPUs must be a positive integer")


def validate_hparams(hparams):
    """
    Limited number of checks performed on the validity of the hyperparameters.
    The file is generally considered to follow the semantics of the
    mpunet.bin.defaults hyperparameter files.

    Args:
        hparams: A YAMLHParams object
    """
    # Tests for valid hparams
    if hparams["fit"].get("class_weights") and hparams["fit"]["loss"] not in \
            ("SparseFocalLoss",):
        # Only currently supported losses
        raise ValueError("Invalid loss function '{}' used with the "
                         "'class_weights' "
                         "parameter".format(hparams["fit"]["loss"]))
    if hparams["fit"]["loss"] == "WeightedCrossEntropyWithLogits":
        if not bool(hparams["fit"]["class_weights"]):
            raise ValueError("Must specify 'class_weights' argument with loss"
                             "'WeightedCrossEntropyWithLogits'.")
        if not hparams["build"]["out_activation"] == "linear":
            raise ValueError("Must use out_activation: linear parameter with "
                             "loss 'WeightedCrossEntropyWithLogits'")
    if not hparams["train_data"]["base_dir"]:
        raise ValueError("No training data folder specified in parameter file.")


def remove_previous_session(project_folder):
    """
    Deletes various mpunet project folders and files from
    [project_folder].

    Args:
        project_folder: A path to a mpunet project folder
    """
    import shutil
    # Remove old files and directories of logs, images etc if existing
    paths = [os.path.join(project_folder, p) for p in ("images",
                                                       "logs",
                                                       "tensorboard",
                                                       "views.npz",
                                                       "views.png")]
    for p in filter(os.path.exists, paths):
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def get_logger(project_dir, overwrite_existing):
    """
    Initialises and returns a Logger object for a given project directory.
    If a logfile already exists at the specified location, it will be
    overwritten if continue_training == True, otherwise raises RuntimeError

    Args:
        project_dir: Path to a mpunet project folder
        overwrite_existing: Whether to overwrite existing logfile in project_dir

    Returns:
        A mpunet Logger object initialized in project_dir
    """
    # Define Logger object
    from mpunet.logging import Logger
    try:
        logger = Logger(base_path=project_dir,
                        print_to_screen=True,
                        overwrite_existing=overwrite_existing)
    except OSError as e:
        raise RuntimeError("[*] A training session at '%s' already exists."
                           "\n    Use the --overwrite flag to "
                           "overwrite." % project_dir) from e
    return logger


def get_gpu_monitor(num_GPUs, logger):
    """
    Args:
        num_GPUs: Number of GPUs to train on
        logger: A mpunet logger object that will be passed to
                the GPUMonitor

    Returns:
        If num_GPUs >= 0, returns a GPUMonitor object, otherwise returns None
    """
    if num_GPUs >= 0:
        # Initialize GPUMonitor in separate fork now before memory builds up
        from mpunet.utils.system import GPUMonitor
        gpu_mon = GPUMonitor(logger)
    else:
        gpu_mon = None
    return gpu_mon


def set_gpu(gpu_mon, args):
    """
    Sets the GPU visibility based on the passed arguments. Takes an already
    initialized GPUMonitor object. Sets GPUs according to args.force_GPU, if
    specified, otherwise sets first args.num_GPUs free GPUs on the system.

    Stops the GPUMonitor process once GPUs have been set
    If gpu_mon is None, this function does nothing

    Args:
        gpu_mon: An initialized GPUMonitor object or None
        args: argparse arguments

    Returns: The number of GPUs that was actually set (different from
    args.num_GPUs if args.force_GPU is set to more than 1 GPU)
    """
    num_GPUs = args.num_GPUs
    if gpu_mon is not None:
        if not args.force_GPU:
            gpu_mon.await_and_set_free_GPU(N=num_GPUs, sleep_seconds=120)
        else:
            gpu_mon.set_GPUs = args.force_GPU
            num_GPUs = len(args.force_GPU.split(","))
        gpu_mon.stop()
    return num_GPUs


def get_data_sequences(project_dir, hparams, logger, args):
    """
    Loads training and validation data as specified in the hyperparameter file.
    Returns a batch sequencer object for each dataset, not the  ImagePairLoader
    dataset itself. The preprocessing function may make changes to the hparams
    dictionary.

    Args:
        project_dir: A path to a mpunet project
        hparams: A YAMLHParams object
        logger: A mpunet logging object
        args: argparse arguments

    Returns:
        train: A batch sequencer object for the training data
        val: A batch sequencer object for the validation data,
             or None if --no_val was specified
        hparams: The YAMLHParams object
    """
    from mpunet.preprocessing import get_preprocessing_func
    func = get_preprocessing_func(hparams["build"].get("model_class_name"))
    hparams['fit']['flatten_y'] = True
    hparams['fit']['max_loaded'] = args.max_loaded_images
    hparams['fit']['num_access'] = args.num_access
    train, val, hparams = func(hparams=hparams,
                               logger=logger,
                               just_one=args.just_one,
                               no_val=args.no_val,
                               continue_training=args.continue_training,
                               base_path=project_dir)
    return train, val, hparams


def get_model(project_dir, train_seq, hparams, logger, args):
    """
    Initializes a tf.keras Model from mpunet.models as specified in
    hparams['build']. If args.continue_training, the best previous model stored
    in [project_dir]/models will be loaded.

    If hparams["build"]["biased_output_layer"] is True, sets the bias weights
    on the final conv. layer so that a zero-input gives an output of class
    probabilities equal to the class frequencies of the training set.

    Args:
        project_dir: A path to a mpunet project folder
        train_seq: A mpunet.sequences object for the training data
        hparams: A mpunet YAMLHParams object
        logger: A mpunet logging object
        args: argparse arguments

    Returns:
        model: The model to fit
        org_model: The original, non-GPU-distributed model
                   (Same as model if num_GPUs==1)
    """
    from mpunet.models import model_initializer
    # Build new model (or continue training an existing one)
    hparams["build"]['flatten_output'] = True
    model = model_initializer(hparams=hparams,
                              continue_training=args.continue_training,
                              project_dir=project_dir,
                              logger=logger)
    # Initialize weights in final layer?
    if not args.continue_training and hparams["build"].get("biased_output_layer"):
        from mpunet.utils.utils import set_bias_weights_on_all_outputs
        set_bias_weights_on_all_outputs(model,
                                        train_seq.image_pair_queue,
                                        hparams,
                                        logger)
    return model


def save_final_weights(model, project_dir, logger=None):
    """
    Saves the weights of 'model' to [project_dir]/model/model_weights.h5

    Args:
        model: A tf.keras Model object
        project_dir: A path to a mpunet project
        logger: mpunet logging object, or None
    """
   
    if not os.path.exists("%s/model" % project_dir):
        os.mkdir("%s/model" % project_dir)
    model_path = "%s/model/model_weights.h5" % project_dir
    if logger:
        logger("Saving current model to: %s" % model_path)
    model.save_weights(model_path)
'''
def get_steps(sequence, im_per_epoch=None):
    """ Returns the number of gradient steps to take in an epoch """
    if im_per_epoch:
        steps = int(np.ceil(im_per_epoch / sequence.batch_size))
    else:
        steps = len(sequence)
    return steps
'''
from mpunet.preprocessing.data_loader import get_train_generators
def train1(cf):
    """Perform training from scratch."""
    # do not use all gpus
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
'''
    # Get number of steps per train epoch
        train_steps = get_steps(train, train_im_per_epoch)
        self.logger("Using %i steps per train epoch (total batches=%i)" %
                    (train_steps, len(train)))

        if val is None:
            # No validation to be performed, remove callbacks that might need
            # validation data to function properly
            remove_validation_callbacks(callbacks, self.logger)
        else:
            val.batch_size = batch_size
            val_steps = get_steps(val, val_im_per_epoch)
            self.logger("Using %i steps per val epoch (total batches=%i)" %
                        (val_steps, len(val)))
            # Add validation callback
            # Important: Should be first in callbacks list as other CBs may
            # depend on the validation metrics/loss
            validation = Validation(val,
                                    steps=val_steps,
                                    ignore_class_zero=val_ignore_class_zero,
                                    logger=self.logger,
                                    verbose=verbose)
            callbacks = [validation] + callbacks
'''  
def save_image_for_PU(data, set):
  #print(data.shape)
  print("length of orginal data is=",len(data))
  for j in range(3):
    X, y, _ = data[j]
    print(X.shape, y.shape)
        #subdir = os.path.join(self.save_path, subdir)
        #if not os.path.exists(subdir):
            #os.mkdir(subdir)

        # Plot each sample in the batch
    for i, (im, lab) in enumerate(zip(X, y)):
            #fig, (ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 6))
      #print("image and label shape before reshape", im.shape, lab.shape)
      lab = lab.reshape(im.shape[:-1] + (lab.shape[-1],))
      #print("image and label shape after reshape", im.shape, lab.shape)
      if im.shape[2]!=3 or im.shape[2]!=4:
        #im=np.squeeze(im, axis = 2)  
        lab=np.squeeze(lab, axis = 2)
      data_format='NCHW'
      img_out_path = os.path.join("/home/sandeep/KU_research/PMPU/output/quarter", set, "cities_name", "image"+"_"+str(j)+"_"+ str(i)+"_leftImg8bit"+".npy")
      lab_out_path = os.path.join("/home/sandeep/KU_research/PMPU/output/quarter", set, "cities_name", "image"+"_"+str(j)+"_"+ str(i)+"_gtFine_labelIds"+".npy")
      img_arr = np.array(im).astype(np.float32)
      lab_arr = np.array(lab).astype(np.uint8)
      channel_axis = 0 if img_arr.shape[0] == 3 else 2
      if data_format == 'NCHW' and channel_axis != 0 and len(img_arr.shape)==3:
        img_arr = np.transpose(img_arr, axes=[2,0,1])
      np.save(img_out_path, img_arr)
      np.save(lab_out_path, lab_arr)
      
def run(project_dir, gpu_mon, logger, args):
    """
    Runs training of a model in a mpunet project directory.

    Args:
        project_dir: A path to a mpunet project
        gpu_mon: An initialized GPUMonitor object
        logger: A mpunet logging object
        args: argparse arguments
    """
    # Read in hyperparameters from YAML file
    from mpunet.hyperparameters import YAMLHParams
    hparams = YAMLHParams(project_dir + "/train_hparams.yaml", logger=logger)
    validate_hparams(hparams)

    # Wait for PID to terminate before continuing?
    if args.wait_for:
        from mpunet.utils import await_PIDs
        await_PIDs(args.wait_for)

    # Prepare sequence generators and potential model specific hparam changes
    train, val, hparams = get_data_sequences(project_dir=project_dir,
                                             hparams=hparams,
                                             logger=logger,
                                             args=args)
    set="train"
    save_image_for_PU(train, set)
    set="val"
    save_image_for_PU(val, set)                                         
    # Set GPU visibility and create model with MirroredStrategy
    set_gpu(gpu_mon, args)
    #import tensorflow as tf
    #with tf.distribute.MirroredStrategy().scope():
        #model = get_model(project_dir=project_dir, train_seq=train,
         #                 hparams=hparams, logger=logger, args=args)
                    
    cf = SourceFileLoader('cf', args.config).load_module()
    if args.data_dir != '':
         cf.data_dir = args.data_dir

    # prepare experiment directory
    if not os.path.isdir(cf.exp_dir):
      os.mkdir(cf.exp_dir)

    # log to file and console
    log_path = os.path.join(cf.exp_dir, 'train.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Logging to {}'.format(log_path))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.disable_v2_behavior
    tf.compat.v1.reset_default_graph()
    train1(cf)
    """
        # Get trainer and compile model
        from mpunet.train import Trainer
        trainer = Trainer(model, logger=logger)
        trainer.compile_model(n_classes=hparams["build"].get("n_classes"),
                              reduction=tf.keras.losses.Reduction.NONE,
                              **hparams["fit"])
    """
    # Debug mode?
    if args.debug:
        from tensorflow.python import debug as tfdbg
        from tensorflow.keras import backend as K
        K.set_session(tfdbg.LocalCLIDebugWrapperSession(K.get_session()))
    """
    # Fit the model
    _ = trainer.fit(train=train, val=val,
                    train_im_per_epoch=args.train_images_per_epoch,
                    val_im_per_epoch=args.val_images_per_epoch,
                    hparams=hparams, no_im=args.no_images, **hparams["fit"])
    """                
    #save_final_weights(model, project_dir, logger)


def entry_func(args=None):
    """
    Function called from mp to init training
    1) Parses command-line arguments
    2) Validation command-line arguments
    3) Checks and potentially deletes a preious version of the project folder
    4) Initializes a logger and a GPUMonitor object
    5) Calls run() to start training

    Args:
        args: None or arguments passed from mp
    """
    # Get and check args
    args = get_argparser().parse_args(args)
    validate_args(args)

    # Check for project dir
    project_dir = os.path.abspath(args.project_dir)
    validate_project_dir(project_dir)
    os.chdir(project_dir)

    # Get project folder and remove previous session if --overwrite
    if args.overwrite:
        remove_previous_session(project_dir)

    # Get logger object. Overwrites previous logfile if args.continue_training
    logger = get_logger(project_dir, args.continue_training)
    logger("Fitting model in path:\n%s" % project_dir)

    # Start GPU monitor process, if num_GPUs > 0
    gpu_mon = get_gpu_monitor(args.num_GPUs, logger)

    try:
        run(project_dir=project_dir, gpu_mon=gpu_mon, logger=logger, args=args)
    except Exception as e:
        if gpu_mon is not None:
            gpu_mon.stop()
        raise e


if __name__ == "__main__":
    entry_func()
