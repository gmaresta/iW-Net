
import json

from models import unet3D


def load_model():

    #read the config file
    with open('config.json', 'r') as f:
        config = json.load(f)  
    config_gen = {}
    
    config_gen['train'] = dict(config['generator'])
    config_gen['train']['IMAGE_FOLDER'] = ""
    config_gen['train']['SEG_FOLDER'] =""
    config_gen['train']['IMAGE_EXTENSION'] = '.npy'
    config_model = dict(config['model'])
    
    unet = unet3D(config_gen['train']['IMAGE_W'], config_gen['train']['IMAGE_W']/4,
                  config_model['INIT_MAPS'], config_model['REGUL'],
                  config_model['DROPOUT'],0,
                  bn=config_model['BATCH_NORM'])
    
    
    
    model_guided = unet.correctionModel('best_1st.hdf5')
    model_guided.load_weights('best.hdf5')
    #model.load_weights(os.path.join(args.exp_path,'best.hdf5'))
    
    
    temp = model_guided.get_layer(name='model_2')
    W = temp.get_weights()
    
    #print(W)
    
    
    
    iterative = unet.createGuidedModel()
    iterative.set_weights(W)
    
    return model_guided