class ProbUNet(snt.Module):
    """Probabilistic U-Net."""

    def __init__(self,
                 latent_dim,
                 num_channels,
                 num_classes,
                 num_1x1_convs=3,
                 nonlinearity=tf.nn.relu,
                 num_convs_per_block=3,
                 initializers={'w': he_normal(), 'b': tf.compat.v1.truncated_normal_initializer(stddev=0.001)},
                 regularizers={'w': tf.keras.regularizers.l2(0.5 * (1.0)), 'b': tf.keras.regularizers.l2(0.5 * (1.0))},
                 data_format='NCHW',
                 down_sampling_op=lambda x, df:\
                         tf.nn.avg_pool2d(input=x, ksize=[1,1,2,2], strides=[1,1,2,2], padding='SAME', data_format=df),
                 up_sampling_op=lambda x, size:\
                         tf.image.resize(x, size, method=tf.image.ResizeMethod.BILINEAR),
                 name='prob_unet'):
        super(ProbUNet, self).__init__(name=name)
        self._data_format = data_format
        self._num_classes = num_classes
        #with self._enter_variable_scope():
        with tf.compat.v1.variable_scope(name):
          self._unet = UNet(num_channels=num_channels, num_classes=num_classes, nonlinearity=nonlinearity,
                              num_convs_per_block=num_convs_per_block, initializers=initializers,
                              regularizers=regularizers, data_format=data_format,
                              down_sampling_op=down_sampling_op, up_sampling_op=up_sampling_op)

          self._f_comb = Conv1x1Decoder(num_classes=num_classes, num_1x1_convs=num_1x1_convs,
                                          num_channels=num_channels[0], nonlinearity=nonlinearity,
                                          data_format=data_format, initializers=initializers, regularizers=regularizers)

          self._prior =\
                AxisAlignedConvGaussian(latent_dim=latent_dim, num_channels=num_channels,
                                        nonlinearity=nonlinearity, num_convs_per_block=num_convs_per_block,
                                        initializers=initializers, regularizers=regularizers, name='prior')

          self._posterior =AxisAlignedConvGaussian(latent_dim=latent_dim, num_channels=num_channels,
                            nonlinearity=nonlinearity, num_convs_per_block=num_convs_per_block,
                                        initializers=initializers, regularizers=regularizers, name='posterior')     
    def __call__(self, x, y, is_training, one_hot_labels): 
    


    def elbo(self, seg, beta=1.0, analytic_kl=True, reconstruct_posterior_mean=False, z_q=None, one_hot_labels=True,
             loss_mask=None):
        """
        Calculate the evidence lower bound (elbo) of the log-likelihood of P(Y|X).
        :param seg: 4D tensor
        :param analytic_kl: bool, if False calculate the KL via sampling
        :param z_q: 4D tensor
        :param one_hot_labels: bool, if False expects integer labeled segmentation of shape N1HW or NHW1
        :param loss_mask: 4D tensor, binary
        :return: 1D tensor
        """
        if z_q is None:
            z_q = self._q.sample()
            #print('z_q valus is =',z_q) 

        self._kl = tf.reduce_mean(input_tensor=self.kl(analytic_kl, z_q))
        self._rec_logits = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, z_q=z_q)
        rec_loss = ce_loss(labels=seg, logits=self._rec_logits, n_classes=self._num_classes,
                           loss_mask=loss_mask, one_hot_labels=one_hot_labels)
        performance = performance_metrics(labels=seg, logits=self._rec_logits, n_classes=self._num_classes,
                           loss_mask=loss_mask, one_hot_labels=one_hot_labels)
        performances=calc_confusion(labels=seg, logits=self._rec_logits, n_classes=self._num_classes,
                           loss_mask=loss_mask, one_hot_labels=one_hot_labels)
        #performances = cal_metrics(confusion_matrix)
        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
      	    sess.run(init)
      	    self._precision= performances['precision']
        self._rec_loss = rec_loss['sum']
        self._rec_loss_mean = rec_loss['mean']
        self._accuracy = performance['accuracy_sum']
        self._accuracy_mean = performance['accuracy_mean']   
        return -(self._rec_loss + beta * self._kl)
