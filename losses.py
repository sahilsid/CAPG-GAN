from light_cnn import LightCNN
import tensorflow as tf

class LossFns():
    def __init__(self):
        lcnn = LightCNN(classes=10133,
                extractor_weights='./Weights/LightCNN/extractor.hdf5')
        X = lcnn.extractor().input 
        Y_conv = lcnn.extractor().layers[-3].output
        Y_fc = lcnn.extractor().layers[-1].output

        self.lcnn_feature_extractor  = tf.keras.Model(X, [Y_conv,Y_fc])
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mae = tf.keras.losses.MeanAbsoluteError()
        
        self.loss_weights = {
            'pix':10,
            'adv1':0.1,
            'adv2':0.1,
            'ip'  :0.02,
            'reg' :1e-4
        }

    def multi_scale_pixel_loss(self,I_b,I_b_fake):
        scales = [(32,32),(64,64),(128,128)]
        loss = tf.reduce_mean(tf.stack([self.mae(tf.image.resize(I_b,[scale[0],scale[1]]),tf.image.resize(I_b_fake,[scale[0],scale[1]])) for scale in scales]))
        return loss   

    def discriminator_loss(self,real_output, fake_output):
        real_loss = self.cross_entropy(0.987*tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(0.07*tf.ones_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def G_loss_(self,fake_output):
        return self.cross_entropy(0.986*tf.ones_like(fake_output), fake_output)

    def identity_preserving_loss(self,I_b,I_b_fake):
        with tf.device('GPU:1'):
            I_b_gray      = tf.image.rgb_to_grayscale(I_b)
            I_b_fake_gray = tf.image.rgb_to_grayscale(I_b_fake)

            conv_features_1,fc_features_1  = self.lcnn_feature_extractor(I_b_gray)
            conv_features_2,fc_features_2  = self.lcnn_feature_extractor(I_b_fake_gray)

            loss = tf.norm(conv_features_1-conv_features_2)+tf.norm(fc_features_1-fc_features_2)

        return loss

    def total_variation_regularization(self,I_b_fake):
        return tf.reduce_sum(tf.image.total_variation(I_b_fake))
    
    
    def generator_loss(self,I_b,I_b_fake,D1_fake,D2_fake):
        
        L1     = self.multi_scale_pixel_loss(I_b,I_b_fake)
        L2     = self.G_loss_(D1_fake)
        L3     = self.G_loss_(D2_fake)
        L4     = self.identity_preserving_loss(I_b,I_b_fake)
        L5     = self.total_variation_regularization(I_b_fake)
        
        total_loss = self.loss_weights['pix']*L1 + self.loss_weights['adv1']*L2 + self.loss_weights['adv2']*L3 + self.loss_weights['ip']*L4 + self.loss_weights['reg']*L5
        
        return total_loss