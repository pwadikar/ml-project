import torch
import torch.nn as nn
from torch.autograd import Variable
#from Data_Loaders import Data_Loaders
class Action_Conditioned_FF(nn.Module):
    def __init__(self,input_size=6,hidden_size=200,output_size=1):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        #self.nonlinear = nn.Sigmoid()
        pass

    def forward(self, input_):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        output = self.fc1(input_)
        output = self.relu(output)
        output = self.fc2(output)
        #output = self.nonlinear(output)
        #print('Prediction',output)
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        #loss_function = nn.MSELoss()
        losses = 0
        
        for idx, sample in enumerate(test_loader):
            #print('This is ',sample['input'])
            input_val = Variable(torch.from_numpy(sample['input']))
            #print('This is ',sample['label'])
            output_val = Variable(torch.tensor(sample['label']))
            #output_val = sample['label']
            network_output = model(input_val.float())
            #print ('This is Label output',output_val)
            #print ('This is network output',network_output)
            loss = loss_function(network_output,output_val)
            losses += loss.item() 
        #print (losses)
        return losses

#def main():
#    model = Action_Conditioned_FF()
#    batch_size = 16
#    data_loaders = Data_Loaders(batch_size)
#    model.evaluate(model,data_loaders.test_loader,nn.MSELoss())

#if __name__ == '__main__':
#    main()
