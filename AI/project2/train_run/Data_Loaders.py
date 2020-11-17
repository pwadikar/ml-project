import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        sensor_row0 = self.data[:,-1]
        sensor_row1 = np.roll(self.data[:,-1],-1)
        sensor_row2 = np.roll(self.data[:,-1],-2)
        #self.data[:,-1]= sensor_row0+sensor_row1
        self.collision_full_data = self.data[self.data[:,6] > 0]
        print('collision data',len(self.collision_full_data))
        print('collision data',self.collision_full_data[0,:])
        dat_min = self.data[:,6]
        print('len',len(dat_min))
        print('len_',int(len(dat_min)/len(self.collision_full_data)))
        len_ = int(len(dat_min)/len(self.collision_full_data))
        for i in range(len_):
            np.random.shuffle(self.collision_full_data)
            self.data = np.append(self.data,self.collision_full_data,axis=0)

        dat_min = self.data[:,6]
        print('len',len(dat_min))
        np.random.shuffle(self.data)

        dat_min = self.data[:,6]
        print('len',len(dat_min))
        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        
        return len(self.data)
        pass

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            return
        #print ('idx=',idx)
        x = self.normalized_data[idx,0:6]
        #print ('X=',x)
        y = self.normalized_data[idx,6]
        #print ('Y=',y)
        return {'input':x, 'label':y}
        
            
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        datlen = len(self.nav_dataset)
        #datlen = 100
        #print('data loader', datlen)
        train_len = int(0.8*datlen)
        test_len = datlen-train_len
        
        self.train_loader, self.test_loader = torch.utils.data.random_split(self.nav_dataset,[train_len,test_len])
        
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary

#def main():
#    batch_size = 16
#    data_loaders = Data_Loaders(batch_size)
#    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
#    for idx, sample in enumerate(data_loaders.train_loader):
#        _, _ = sample['input'], sample['label']
#    for idx, sample in enumerate(data_loaders.test_loader):
#        _, _ = sample['input'], sample['label']

#if __name__ == '__main__':
#    main()
