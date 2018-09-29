import os
from daveglobals import Globals
from shutil import copyfile
from mimetypes import guess_extension
import uuid
import base64
import pickle

def get_person_folder_path(model_path, name):
    folder_name = name.replace(' ', '_')
    folder_path = os.path.join(model_path, "data", folder_name)
    
    return folder_name, folder_path

def load_model(model_path):
    with open(model_path, 'rb') as infile:
        (model, class_names, emb_array, labels) = pickle.load(infile)
        
    print('Loaded classifier model from file "%s"' % model_path)
    
    return model, class_names, emb_array, labels

def save_model(model_file, model, class_names, emb_array, labels):
    with open(model_file, 'wb') as outfile:
        pickle.dump((model, class_names, emb_array, labels), outfile)
        
    print('Saved classifier model to file "%s"' % model_file)

def save_temp_face(image):
    temp_data_path = os.path.join(Globals.temp_path, "data")
    file_path = None
    
    if not os.path.exists(Globals.data_path):
        return None, "Training Data Path not found"
    
    if not os.path.exists(Globals.temp_path):
        os.makedirs(Globals.temp_path)

    if not os.path.exists(temp_data_path):
        os.makedirs(temp_data_path)
    
    try:
        if os.path.isfile(image):
            print("Copy file to temp path")
            f_name = os.path.basename(image)
            file_path = os.path.join(temp_data_path, f_name)
            copyfile(image, file_path)
    
    except ValueError:
        print("Convert base64 image to temp file")
    
        extension = None
        
        if image[0:len("data:")] == "data:":
            # wb... 'w' = open for writing, 'b' = binary mode
            image_split = image.split(',', 1)
            
            prefix = image_split[0]
            prefix = prefix[len("data:"):len(prefix) - len("base64,")]
            extension = guess_extension(prefix)
            image = image_split[1]
        else:
            # Must have been created by us, so we know it's a .png
            extension = ".png"
    
        if extension == None:
            return None, "Not a valid file mime type"
    
        file_guid = str(uuid.uuid4())
        file_name = file_guid + extension
        file_path = os.path.join(temp_data_path, file_name)
        imgdata = base64.b64decode(image)
        
        with open(file_path, "wb") as fh:
            fh.write(imgdata)
            
        if imgdata == None:
            return None, "No image data available"
            
    return file_path, ""
