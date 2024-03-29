import pickle
import os

class Logger:

    def set_default_filename(self, filename):
        self.default_filename = filename

    def create_session_folder(self, path):
        try:  
            os.makedirs(path)
        except OSError:  
            print ("Creation of the directory %s failed" % path)
        else:  
            print ("     =====> Successfully created the directory %s \n" % path)

        try:
            os.makedirs(path + "models/")
        except OSError:  
            print ("Creation of the model directory failed")
        else:  
            print ("     =====> Successfully created the model directory")


    def log(self, text):
        with open(self.default_filename, 'a') as f:
            f.writelines(text)
            f.write("\n")

    def save_model(self, model, filename):
        pickle.dump(model, open(filename, 'wb'))
    
    