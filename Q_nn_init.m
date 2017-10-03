function [ net_Q ] = Q_nn_init( Q_NET_SIZES,STATE_SIZE,ACTION_SIZE )
%Q_NN_INIT 
%   initialize 2-hidden-layer Q neural network for the DQN algorithm
 
% load Q
% net_Q=Q;
 
%WeightsPerLayer is the number of neurons in our 2 hidden layer nn
%********************************define and init Q nn********************
net_Q=feedforwardnet(Q_NET_SIZES);
 
% X=[1,0.5,-1,-0.5,1;1,-1,1,0.5,1;1,0.5,-1,-0.5,1;1,0.5,-1,-0.5,1;1,0.5,-1,-0.5,1];   %X is the input and determines the input dimension of the nn
X=rand([STATE_SIZE,STATE_SIZE]);
% T=[1,0,1,0,1;0,1,0,1,0;1,0,1,0,1;1,0,1,0,1;1,0,1,0,1];  %This is for 5 actions 
T=rand([ACTION_SIZE,STATE_SIZE]);
% T=[1,0,1,0,1;0,1,0,1,0;1,0,1,0,1];  %This is for 3 actions  %T is the output and determines the output dimension of the nn.
% FIXME: I tried to use one hot vectors for classification but It doesn't seem to
% work because I'm getting bad outputs
 
% FIXME: initialize smarter (X,T doesn't really make sense)
net_Q.trainFcn = 'traingdm';          %Gradient Descent Backpropagation
net_Q.trainParam.epochs = 1;           %1 epoch becuase we are batch-training
net_Q.trainParam.lr = 0.00025;
net_Q.trainParam.mc = 0.99;
net_Q.trainParam.showWindow=false;    %show GUI for training results
net_Q.divideFcn='dividetrain';       %assign all examples to training (and not to validation/testing)
net_Q.performFcn='mse';             %mean square error performance function
%net_Q.performFcn='crossentropy';     %mean square error performance function
init(net_Q);                        %initialize net
net_Q=train(net_Q,X,T);
 
 
end
