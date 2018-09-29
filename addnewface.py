from Helpers import helpers
from daveglobals import Globals
from mygraph import MyGraph
import shutil
import classifier
import os
import numpy as np
from shutil import copyfile

def add(image, model_folder, name):
    # Create temp image
    file_path, error = helpers.save_temp_face(image)
    
    if error != "":
        return False, error
    
    # Get facenet embeddings
    model_path = os.path.join(Globals.model_path, model_folder)
    classifier_file = os.path.join(model_path, "classifier.pkl")
    
    features = classifier.get_features(Globals.temp_path, MyGraph(), classifier_file)
    
    if features.success == False:
        return False, features.error
    
    # Load model
    (model, class_names, emb_array, labels) = helpers.load_model(features.classifier_filename_exp)
    
    print(emb_array.shape)
    print(features.emb_array.shape)
    emb_array = np.append(emb_array, features.emb_array, axis = 0)
    
    # Add new embedding to array
    print("Emb array")
    print(emb_array.shape)

    matches = next((n for n in class_names if n.lower() == name.lower()), None)
    
    if matches == None:
        print("Name not found... adding new name")
        class_names.append(name)
        
    name_index = class_names.index(name)
    labels.append(name_index)

    folder_name, folder_path = helpers.get_person_folder_path(model_path, name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_name = os.path.basename(file_path)
    copyfile(file_path, os.path.join(folder_path, file_name))
    
    print(len(labels))
    print(len(class_names))
    
    # Retrain
    model.fit(emb_array, labels)
    
    # Save the new model / embeddings etc
    helpers.save_model(features.classifier_filename_exp, model, class_names, emb_array, labels)

    # Cleanup
    shutil.rmtree(Globals.temp_path)
    
    return True, ""