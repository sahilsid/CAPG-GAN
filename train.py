from model import *
from losses import *
from dataset import *
import tqdm

DATASET_DIR          = './Dataset/columbia'
PROCESSED_DATA_DIR   ='./Data'
IMG_NAME_TYPE        = '{}_2m_{}P_0V_0H.jpg'
IMG_DIM              = (128,128)
BATCH_SIZE           = 32
process_dir          = False

data_generator = generate_dataset(DATASET_DIR,PROCESSED_DATA_DIR,IMG_NAME_TYPE,IMG_DIM,BATCH_SIZE,process_dir)
G       = build_generator()
D1      = build_discriminator(1)
D2      = build_discriminator(2)

loss = LossFns()

EPOCHS   = 50
STEPS    = 100
LR       = 2e-4

G_optimizer = tf.keras.optimizers.Adam(LR)
D1_optimizer = tf.keras.optimizers.Adam(LR)
D2_optimizer = tf.keras.optimizers.Adam(LR)

outer_bar = tqdm.tqdm(total=EPOCHS, desc='Epochs', position=1)
for epoch in range(1,EPOCHS+1):
    
    inner_bar = tqdm.tqdm(total=STEPS, desc='Steps', position=0)
    for step in range(STEPS):
        I_a,I_b,P_a,P_b =  next(data_generator)
        with tf.GradientTape(persistent=True) as d_tape:
            I_b_fake = G([I_a,P_a,P_b])
            
            D1_real  = D1([I_b,I_a])
            D1_fake  = D1([I_b_fake,I_a])            
            
            D2_real  = D2([I_b,P_b])
            D2_fake  = D2([I_b_fake,P_b])
            
            G_loss   = loss.generator_loss(I_b,I_b_fake,D1_fake,D2_fake)
            D1_loss  = loss.discriminator_loss(D1_real,D1_fake)
            D2_loss  = loss.discriminator_loss(D2_real,D2_fake)
        
        D1_grad = d_tape.gradient(D1_loss,D1.trainable_variables)
        D2_grad = d_tape.gradient(D2_loss,D2.trainable_variables)
        
        D1_optimizer.apply_gradients(zip(D1_grad,D1.trainable_variables))
        D2_optimizer.apply_gradients(zip(D2_grad,D2.trainable_variables))
        
        with tf.GradientTape() as g_tape:
            I_b_fake = G([I_a,P_a,P_b])
            D1_fake  = D1([I_b_fake,I_a])            
            D2_fake  = D2([I_b_fake,P_b])
            
            G_loss   = loss.generator_loss(I_b,I_b_fake,D1_fake,D2_fake)
            
        G_grad = g_tape.gradient(G_loss,G.trainable_variables)
        
        G_optimizer.apply_gradients(zip(G_grad,G.trainable_variables))
        
        inner_bar.update(1)
    
    outer_bar.update(1)