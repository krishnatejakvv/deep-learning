import csv, cv2, pickle, multiprocessing, mtcnn, warnings
import numpy as np

# list of age ranges
age_list =  ['(0, 2)',
             '(4, 6)',
             '(8, 12)',
             '(15, 20)',
             '(25, 32)',
             '(38, 43)',
             '(48, 53)',
             '(60, 100)',
            ]
# list of gender
gender_list = ['f', 'm']
    
def process_fold_aligned(fold_id):
    """ 
    read in aligned data
    export aligned data in pkl file
    """
    # read metadata
    fold_data = []   # placeholder for data, will be converted to 4d array
    fold_age = []
    fold_gender = []
    with open ('fold_{0}_data.txt'.format(fold_id)) as file:
        next(file) # skip header
        for line in csv.reader(file, dialect = 'excel-tab'):
            user_id = line[0]
            original_image = line [1]
            face_id = line[2]
            filename = 'faces/' + user_id + '/coarse_tilt_aligned_face.' + face_id + '.' + original_image
            print('non-crop ' + filename, end = ' ... ')
            data = cv2.imread(filename)
            age = line[3]
            gender = line[4]
            if (age in age_list) and (gender in gender_list):
                print('GOOD LABEL')
                fold_age.append(age)
                fold_gender.append(gender)
                data = cv2.resize(data, (816, 816))
                fold_data.append(data)
            else:
                print('NO LABEL')

    fold_data = np.stack(fold_data, axis = 0)
    fold_age = np.array(fold_age)
    fold_gender = np.array(fold_gender)
    # save data
    print('processed data/fold_{0}_data.pkl'.format(fold_id))
    with open('processed data/fold_{0}_data.pkl'.format(fold_id), 'wb') as file:
        pickle.dump([fold_age, fold_gender, fold_data], file, protocol = 4)

def process_fold_aligned_crop(fold_id):
    """
    read in aligned data
    use MTCNN to crop face and resize to (300, 300)
    export cropped face
    """
    # read metadata
    fold_data = []   # placeholder for data, will be converted to 4d array
    fold_age = []
    fold_gender = []
    fold_data_nolabel = []

    detector = mtcnn.MTCNN() # face detector
    with open ('fold_{0}_data.txt'.format(fold_id)) as file:
        next(file) # skip header
        for line in csv.reader(file, dialect = 'excel-tab'):
            user_id = line[0]
            original_image = line [1]
            face_id = line[2]
            filename = 'faces/' + user_id + '/coarse_tilt_aligned_face.' + face_id + '.' + original_image
            print('fold ' + str(fold_id) + ' crop ' + filename, end = ' ... ')
            data = cv2.imread(filename)
            if data is None:
                warnings.warn('CANNOT FIND FILE')
                continue

            # crop face
            faces = detector.detect_faces(data)
            age = line[3]
            gender = line[4]
            if (age in age_list) and (gender in gender_list):
                print('GOOD LABEL ... ', end = '')
                if len(faces) > 0:
                    print('CROPPING SUCCESS')
                    x, y, w, h = faces[0]['box']   # only use the first face
                    x = max(0,x); y = max(0,y)
                    crop_img = data[y:y+h, x:x+w]
                    # resize
                    crop_img = cv2.resize(crop_img, (224, 224))
                    crop_img = cv2.cvtColor(crop_img, cv2.cv2.COLOR_BGR2RGB)
                    # save
                    fold_age.append(age)
                    fold_gender.append(gender)
                    fold_data.append(crop_img)
                else:
                    print('CROPPING FAIL')
            else:
                print('NO LABEL ... ', end = '')
                if len(faces) > 0:
                    print('CROPPING SUCCESS')
                    x, y, w, h = faces[0]['box']   # only use the first face
                    x = max(0,x); y = max(0,y)
                    crop_img = data[y:y+h, x:x+w]
                    # resize
                    crop_img = cv2.resize(crop_img, (224, 224))
                    crop_img = cv2.cvtColor(crop_img, cv2.cv2.COLOR_BGR2RGB)
                    # save
                    nolabel_data.append(crop_img)
                else:
                    print('CROPPING FAIL')
    print(fold_data)
    fold_data = np.stack(fold_data, axis = 0)
    fold_age = np.array(fold_age)
    fold_gender = np.array(fold_gender)

    # save data
    print('processed data/fold_{0}_data_cropped.pkl'.format(fold_id))
    with open('processed data/fold_{0}_data_cropped.pkl'.format(fold_id), 'wb') as file:
        pickle.dump([fold_age, fold_gender, fold_data], file, protocol = 4)


if __name__ == '__main__':
#    for fold_id in range(5):
#        print('non-crop fold = ' + str(fold_id))
#        process_fold_aligned(fold_id)
    global nolabel_data
    nolabel_data = []
    for fold_id in range(5):
        print('crop fold = ' + str(fold_id))
        process_fold_aligned_crop(fold_id)

    nolabel_data = np.stack(nolabel_data, axis = 0)
    with open('processed data/nolabel_cropped.pkl'.format(fold_id), 'wb') as file:
        pickle.dump(nolabel_data, file, protocol = 4)
