from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF
from torch.autograd import Variable
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 1
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001) 
    loss_function=nn.MSELoss(reduction='sum')
    #loss_function=nn.BCELoss()
    losses = []
    #min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    #losses.append(min_loss)
    
    temp_op=[]
    temp_op.append(0.0)
    for epoch_i in range(no_epochs):
        loss1 =0
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): 
            # sample['input'] and sample['label']
            #print('Sample', sample['input'])
            input_val = Variable(torch.from_numpy(sample['input']))
            temp_op[0] = sample['label']
            output_val = Variable(torch.tensor(temp_op))
            optimizer.zero_grad()
            
            network_output = model(input_val.float())
            #print('SIZE1 ',network_output[0].item())
            #print('SIZE2 ',output_val[0])
            
            loss = loss_function(network_output.float(),output_val.float())
            #loss = loss_function(network_output,output_val)
            loss.backward()
            optimizer.step()
            loss1+= loss.item()
        print ('Epoch %d, Loss: %.4f' %(epoch_i+1, loss.item()))
        #print('len ', len(data_loaders.train_loader))
        pass
    torch.save(model.state_dict(),"saved/saved_model.pkl",_use_new_zipfile_serialization=False)

    loss1 =0
    for idx, sample in enumerate(data_loaders.test_loader): 
        input_val = Variable(torch.from_numpy(sample['input']))
        temp_op[0] = sample['label']
        output_val = Variable(torch.tensor(temp_op))
        network_output = model(input_val.float())
        if ((output_val.item() - network_output.item()>0.3)):
            loss1 = loss1+1
            print('Error with idx',idx,output_val.item(),network_output.item())            
            print('data',input_val)
        print('done with idx',idx,output_val.item(),network_output.item())    
    print('total test sample',len(data_loaders.test_loader))
    print('loss',loss1)




if __name__ == '__main__':
    no_epochs = 25
    train_model(no_epochs)
    
