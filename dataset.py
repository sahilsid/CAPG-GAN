import os
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
from matplotlib.pyplot import cm


def generate_from_dirs(generator,dir1,dir2,dir3,dir4, batch_size, img_height,img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (img_height,img_width),
                                          batch_size = batch_size,
                                          shuffle=False, 
                                          seed=7)

    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (img_height,img_width),
                                          batch_size = batch_size,
                                          shuffle=False, 
                                          seed=7)
    genX3 = generator.flow_from_directory(dir3,
                                          target_size = (img_height,img_width),
                                          batch_size = batch_size,color_mode='grayscale',
                                          shuffle=False, 
                                          seed=7)

    genX4 = generator.flow_from_directory(dir4,
                                          target_size = (img_height,img_width),
                                          batch_size = batch_size,color_mode='grayscale',
                                          shuffle=True, 
                                          seed=7)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            X4i = genX4.next()
            yield X1i[0], X2i[0],X3i[0], X4i[0]  



def generate_dataset(DATASET_DIR = './Dataset/columbia',PROCESSED_DATA_DIR='./Data',IMG_NAME_TYPE = '{}_2m_{}P_0V_0H.jpg',IMG_DIM = (128,128),batch_size=128,raw=False):

    if not os.path.exists(PROCESSED_DATA_DIR+'/source/1'):
        os.makedirs(PROCESSED_DATA_DIR+'/source/1')

    if not os.path.exists(PROCESSED_DATA_DIR+'/target/1'):
        os.makedirs(PROCESSED_DATA_DIR+'/target/1')

    if not os.path.exists(PROCESSED_DATA_DIR+'/source_pose_embedding/1'):
        os.makedirs(PROCESSED_DATA_DIR+'/source_pose_embedding/1')


    if not os.path.exists(PROCESSED_DATA_DIR+'/target_pose_embedding/1'):
        os.makedirs(PROCESSED_DATA_DIR+'/target_pose_embedding/1')

    
    if(raw):   
        source_dir = os.path.join(PROCESSED_DATA_DIR,'source/1')
        target_dir = os.path.join(PROCESSED_DATA_DIR,'target/1')
        source_pose_dir = os.path.join(PROCESSED_DATA_DIR,'source_pose_embedding/1')
        target_pose_dir = os.path.join(PROCESSED_DATA_DIR,'target_pose_embedding/1')

        detector = MTCNN()

        individuals = [os.path.join(DATASET_DIR, o) for o in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR,o))]

        pose_angles = ['15','30','-15','-30']

        for individual in individuals :
            individual_id = individual.split('/')[-1]
            poses = os.listdir(individual)

            frontal_face_path =  os.path.join(individual,IMG_NAME_TYPE.format(individual_id,0))
            frontal_face      =  cv2.cvtColor(cv2.imread(frontal_face_path),cv2.COLOR_BGR2RGB)
            frontal_face      =  cv2.resize(frontal_face, IMG_DIM, interpolation = cv2.INTER_AREA)

            frontal_landmark  = detector.detect_faces(frontal_face)

            if(len(frontal_landmark)>0):
                frontal_landmark  = frontal_landmark[0]

                frontal_mask      = 1.0*np.zeros_like(frontal_face[:,:,0])

                for x,y in frontal_landmark['keypoints'].values():
                    frontal_mask[x,y] = 1.0

                frontal_mask = gaussian_filter(frontal_mask,sigma=2)

                for pose in pose_angles:

                    img_name = IMG_NAME_TYPE.format(individual_id,pose)
                    profile_face_path =  os.path.join(individual,img_name)
                    profile_face      =  cv2.cvtColor(cv2.imread(profile_face_path),cv2.COLOR_BGR2RGB)
                    profile_face      =  cv2.resize(profile_face, IMG_DIM, interpolation = cv2.INTER_AREA)

                    profile_landmark  = detector.detect_faces(profile_face)
                    if(len(profile_landmark)>0):
                        profile_landmark  = profile_landmark[0]
                        profile_mask      = 1.0*np.zeros_like(profile_face[:,:,0])

                        for x,y in profile_landmark['keypoints'].values():
                            profile_mask[x,y] = 1.0

                        profile_mask = gaussian_filter(profile_mask,sigma=2)

                        choice_prob = np.random.uniform()
                        if(choice_prob>0.5):
                            plt.imsave('{}/{}'.format(source_dir,img_name),frontal_face)
                            plt.imsave('{}/{}'.format(target_dir,img_name),profile_face)
                            plt.imsave('{}/{}'.format(source_pose_dir,img_name),frontal_mask,cmap=cm.gray)
                            plt.imsave('{}/{}'.format(target_pose_dir,img_name),profile_mask,cmap=cm.gray)
                        else:
                            plt.imsave('{}/{}'.format(source_dir,img_name),profile_face)
                            plt.imsave('{}/{}'.format(target_dir,img_name),frontal_face)
                            plt.imsave('{}/{}'.format(source_pose_dir,img_name),profile_mask,cmap=cm.gray)
                            plt.imsave('{}/{}'.format(target_pose_dir,img_name),frontal_mask,cmap=cm.gray)

    source_dir = os.path.join(PROCESSED_DATA_DIR,'source')
    target_dir = os.path.join(PROCESSED_DATA_DIR,'target')
    source_pose_dir = os.path.join(PROCESSED_DATA_DIR,'source_pose_embedding')
    target_pose_dir = os.path.join(PROCESSED_DATA_DIR,'target_pose_embedding')
                         
    input_imgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
    datagenerator=generate_from_dirs(generator=input_imgen,
                                               dir1=source_dir,
                                               dir2=target_dir,
                                               dir3=source_pose_dir,
                                               dir4=target_pose_dir,
                                               batch_size=batch_size,
                                               img_height=IMG_DIM[0],
                                               img_width=IMG_DIM[1])      
    return datagenerator

# generate_dataset(raw=True)
