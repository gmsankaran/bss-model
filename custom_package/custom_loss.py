import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class AsymmetricHybridLoss(Loss):
    def __init__(self, alpha=0.5, lambda_bias=1.0, lambda_over=1.0, lambda_under=0.25, local_corr=True, name='asymmetric_hybrid_loss', reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, **kwargs):
        super(AsymmetricHybridLoss, self).__init__(name=name, reduction=reduction, **kwargs)
        self.alpha = alpha
        self.lambda_bias = lambda_bias
        self.lambda_over = lambda_over
        self.lambda_under = lambda_under
        self.local_corr = local_corr

    def call(self, Y_true, Y_pred):
        errors = Y_pred - Y_true
        over_predictions = tf.maximum(errors, 0)
        under_predictions = tf.minimum(errors, 0)

        mae = tf.reduce_mean(self.lambda_over * tf.abs(over_predictions) + self.lambda_under * tf.abs(under_predictions), axis=-1)
        mse = tf.reduce_mean(tf.square(errors), axis=-1)

        loss = self.alpha * mse + (1 - self.alpha) * mae

        if self.local_corr:
            bias_correction = tf.reduce_mean(Y_pred, axis=-1) - tf.reduce_mean(Y_true, axis=-1)
        else:
            bias_correction = tf.reduce_mean(Y_pred) - tf.reduce_mean(Y_true)
        loss += self.lambda_bias * tf.square(bias_correction)

        return loss

    def get_config(self):
        config = super(AsymmetricHybridLoss, self).get_config()
        config.update({
            'alpha': self.alpha,
            'lambda_bias': self.lambda_bias,
            'lambda_over': self.lambda_over,
            'lambda_under': self.lambda_under,
            'local_corr': self.local_corr
        })
        return config
