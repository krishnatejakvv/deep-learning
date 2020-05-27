import pickle
print("""
----- USAGE INFO -------
make sure you have the folder name `processed data` and 5 files inside of it
The data has 5 folds, with fold_id from 0 to 4
This is first set of data for our DL models, it contains RGB of aligned images
all the function fetch_data_aligned(fold_id) to get the data from a fold_id
the data contains:
        age: age labels, 1D numpy array
        gender: gender labels, 1D numpy array
        RGB: 4D numpy array: 
            D1: picture id
            D2-D3: vertical and horizontal pixel id
            D4: channel
            value: RGB value     
For example:
    age, gender, RGB_data = fetch_data_aligned(0)
    this will return the data for fold 0
    
WARNING: make sure you have at least 8GB on your computer, the data is quite big

RGB DATA FOR CROPPED FACE ONLY:
    age, gender, RGB_data = fetch_data_cropped(fold_id)
""")



def fetch_data_aligned(fold_id):
    """ return:
        age: age labels, 1D numpy array
        gender: gender labels, 1D numpy array
        RGB: 4D numpy array: 
            D1: picture id
            D2-D3: vertical and horizontal pixel id
            D4: channel
            value: RGB value     
    """
    with open('processed data/fold_{0}_data.pkl'.format(fold_id), 'rb') as file:
        age, gender, RGB = pickle.load(file)
    return age, gender, RGB

def fetch_data_cropped(fold_id):
    """ return:
        age: age labels, 1D numpy array
        gender: gender labels, 1D numpy array
        RGB: 4D numpy array: 
            D1: picture id
            D2-D3: vertical and horizontal pixel id (300*300)
            D4: channel
            value: RGB value     
    """
    with open('cropped_data/fold_{0}_data_cropped.pkl'.format(fold_id), 'rb') as file:
        age, gender, RGB = pickle.load(file)
    return age, gender, RGB