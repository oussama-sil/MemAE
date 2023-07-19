import numpy as np
import os
from PIL import Image # For loading images 
import tifffile
import matplotlib.pyplot as plt
import random 
from skimage.transform import resize

np.set_printoptions(threshold=np.inf)  # Display all elements in the console 

# Function to load trainning data from the UCSD dataset 
# Dataset output size: m * NB_Frame * H * W * 1
def load_data_train(H,W, load_ped2 = True ,load_ped1=False,path_ped2='./UCSDped2',path_ped1='./UCSDped1',train=True):
    """
        Output :dataset stored as dictionary where keys are the name of the videos and the values are arrays containing the list of frames in each video  
    """
    images = {}
    #? Loading Ped2
    if load_ped2:
        print("Loading dataset PED2")
        # Path to trainning videos 
        if train:
            path_train = path_ped2+'/Train/'
        else:
            path_train = path_ped2+'/Test/'
        # list of directories containning the sequences
        dirs_img = [name for name in os.listdir(path_train) if os.path.isdir(os.path.join(path_train, name))]
        for dir in dirs_img:        # Loop over the sub directories 
            print("Loading : "+dir,end=" : ")
            images[dir] = []
            for filename in os.listdir(path_train+dir):
                if filename.endswith(".tif") :
                    # Construct the full file path
                    file_path = path_train+dir+'/'+filename
                    # Open the image file
                    image = tifffile.imread(file_path)
                    # Resizing the image
                    image = resize(image, (H, W,1))

                    images[dir].append(np.array(image))

                elif filename.endswith(".bmp") :
                    # Construct the full file path
                    file_path = path_train+dir+'/'+filename
                    # Open the image file
                    image = Image.open(file_path)
                    # Resizing the image
                    image = image.resize((H, W))

                    images[dir].append(np.array(image)[:,:,np.newaxis])
            print(f"Loaded {len(images[dir])} images")
    return images


# For loading the trainning dataset 
def prepare_data_train_memae(videos,nb_frames,shuffle=False,m=154,save=False,save_path='./'):
    """
        Output : dataset as numpay array of shape (m,nb_frames,H,W,1)
    """
    data= []
    # Divide the videos to sequences of length nb_frames
    for video in videos.values() :
        nb = int(len(video) / nb_frames)
        for i in range(nb):
            seq = video[i*nb_frames:i*nb_frames+nb_frames]
            data.append(seq)

    # If shuffle : frames can appear in many sequences (Data augmentation)
    if shuffle:
        while len(data)<m:
            # Pick random video 
            random_video = random.choice(list(videos.values()))
            # Pick random starting frame 
            random_indx = random.randint(0,len(random_video)-nb_frames)
            data.append(random_video[random_indx:random_indx+nb_frames])
    data = np.array(data)

    # Save the loaded data to .npy file
    if save:
        np.save(save_path+'/train_'+str(data.shape)+'.npy', data)
        print(f"Data of shape {data.shape} saved")

    return data

def create_train_dataset_memae(H,W,nb_frames,load_ped2 = True ,load_ped1=False,shuffle=False,m=154,save=True,save_path='.',path_ped2='./UCSDped2',path_ped1='./UCSDped1'):
    path = save_path+'/train_'+str((m,nb_frames,H,W,1))+'.npy'
    print("\n<<<  Loading trainning dataset >>>")
    print("path : "+path)
    if os.path.exists(path):
        print("Dataset alreardy prepared, loading it")
        data = np.load(path)
    else:
        print("Preparing and Loading the dataset")
        videos = load_data_train(H,W,load_ped2 = load_ped2 ,load_ped1=load_ped1,path_ped2=path_ped2,path_ped1=path_ped1)
        data = prepare_data_train_memae(videos,nb_frames,shuffle=shuffle,save=save,save_path=save_path,m=m)
    print(f"Data loaded , shape {data.shape}, size {data.nbytes / (1024 * 1024)} MB \n")
    return data



# For loading testing dataset 
def prepare_data_test_memae(videos,nb_frames,save=False,save_path='./'):
    """
        Output : dataset as numpay array of shape (m,nb_frames,H,W,1), ground truth for each sequence (m,nb_frames,H,W,1), labeling of the dataset 
    """
    data= []
    data_gt = []
    data_labels = []
    # Divide the videos to sequences of length nb_frames
    for video_name in videos :
        video = videos[video_name]
        if "_gt" in video_name:
            nb = int(len(video) / nb_frames)
            for i in range(nb):
                seq = video[i*nb_frames:i*nb_frames+nb_frames]
                data_gt.append(seq)
                # Labeling 
                labels = []
                for se in seq:
                    if 1 in se: # Case abnormal event 
                        labels.append(0) # Abnormal event 
                    else :
                        labels.append(1) # Normal event
                data_labels.append(labels)
        else:
            nb = int(len(video) / nb_frames)
            for i in range(nb):
                seq = video[i*nb_frames:i*nb_frames+nb_frames]
                data.append(seq)

    data = np.array(data)
    data_gt = np.array(data_gt)
    data_labels = np.array(data_labels)


    # # Save the loaded data to ...npy files
    if save:
        np.save(save_path+'/test_data_'+str(data.shape)+'.npy', data)
        np.save(save_path+'/test_data_gt_'+str(data.shape)+'.npy', data_gt)
        np.save(save_path+'/test_data_labels_'+str(data.shape)+'.npy', data_labels)


    return data,data_gt,data_labels

def create_test_dataset_memae(H,W,nb_frames,m=122,load_ped2 = True ,load_ped1=False,save=False,save_path='.',path_ped2='./UCSDped2',path_ped1='./UCSDped1'):
    path_data = save_path+'/test_data_'+str((m,nb_frames,H,W,1))+'.npy'
    path_data_gt = save_path+'/test_data_gt_'+str((m,nb_frames,H,W,1))+'.npy'
    path_data_labels = save_path+'/test_data_labels_'+str((m,nb_frames,H,W,1))+'.npy'
    
    print("\n<<<  Loading testing dataset >>>")
    print("path data : "+path_data)
    print("path data gt : "+path_data_gt)
    print("path data labels: "+path_data_labels)

    if os.path.exists(path_data) and os.path.exists(path_data_gt) and os.path.exists(path_data_labels) :
        print("Dataset alreardy prepared, loading it")
        data = np.load(path_data)
        data_gt = np.load(path_data_gt)
        data_labels = np.load(path_data_labels)

    else:
        print("Preparing and Loading the dataset")
        data = load_data_train(H,W,load_ped2 = load_ped2 ,load_ped1=load_ped1,path_ped2=path_ped2,path_ped1=path_ped1,train=False)
        data,data_gt,data_labels = prepare_data_test_memae(data,nb_frames,save=save,save_path=save_path)
    print(f"Data loaded , shape  data {data.shape}, gt data {data_gt.shape}, labels {data_labels.shape} size {(data_labels.nbytes+data_gt.nbytes+data_labels.nbytes) / (1024 * 1024)} MB \n")
    return data,data_gt,data_labels


if __name__=='__main__':
    #? To create a trainning dataset from the UCSDPed2 dataset
    # data = create_train_dataset_memae(256,256,16,load_ped2 = True ,load_ped1=False,shuffle=False,m=154,save=True,save_path='.')

    #? To create a trainning dataset 
    # data = load_data_train(256,256,load_ped2 = True ,load_ped1=False,path_ped2='./UCSDped2',path_ped1='./UCSDped1',train=False)     
    # data,data_gt,data_labels = prepare_data_test_memae(data,16,save=True,save_path='./')
    data,data_gt,data_labels = create_test_dataset_memae(256,256,16,load_ped2 = True ,load_ped1=False,save=True,save_path='.')

    print(data.shape)
    print(data_gt.shape)
    print(data_labels.shape)

    plt.imshow(data[5,4,:,:],cmap='gray')
    plt.show()
    
    plt.imshow(data_gt[5,4,:,:],cmap='gray')
    plt.show()

    print(data_labels[5,4])
    print(1 in data_gt[5,4,:,:])
    print(data_labels[0,0])
    print(1 in data_gt[0,0,:,:])


    # plt.imshow(data["Test002_gt"][100],cmap='gray')
    # plt.show()