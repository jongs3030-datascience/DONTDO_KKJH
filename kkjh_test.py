import os
import glob
import argparse
import matplotlib

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from kkjh_layers import BilinearUpSampling2D
from kkjh_utils import predict, load_images
from matplotlib import pyplot as plt
import cv2####

def depth_test( inputs = 'examples/*', model = 'nyu.h5',  output = 'results/'):
    # Argument Parser
    """
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
    parser.add_argument('--input', default='examples/*', type=str, help='Input filename or folder.')
    parser.add_argument('--output', default='results/', type=str, help='Output folder path')#추가함 구글 경로에 results폴더 만들어야
    args = parser.parse_args()
    """

    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    #print('Loading model...')

    # Load model into GPU / CPU
    model = load_model(model, custom_objects=custom_objects, compile=False)

    #print('\nModel loaded ({0}).'.format(model))

    # Input images
    #input_folder = glob.glob(input)###
    #inputs = load_images( input_folder )###
    #print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = predict(model, inputs)

        
        
        
    # Display results
    #viz = display_images(outputs.copy(), inputs.copy())
    # plt.figure(figsize=(10,5))
    # plt.imshow(viz)
    # plt.savefig('test.png')
    # plt.show()

    return outputs
