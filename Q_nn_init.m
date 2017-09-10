function [ net_Q ] = Q_nn_init( WeightsPerLayer )
%Q_NN_INIT 
%   initialize 2-hidden-layer Q neural network for the DQN algorithm

% load Q
% net_Q=Q;

%WeightsPerLayer is the number of neurons in our 2 hidden layer nn
%********************************define and init Q nn********************
net_Q=feedforwardnet([WeightsPerLayer,WeightsPerLayer,40]);

X=[1,0.5,-1,-0.5,1;1,-1,1,0.5,1;1,0.5,-1,-0.5,1;1,0.5,-1,-0.5,1];   %X is the input and determines the input dimension of the nn
% T=[1,0,1,0,1;0,1,0,1,0;1,0,1,0,1;1,0,1,0,1;1,0,1,0,1];  %This is for 5 actions 
 T=[1,0,1,0,1;0,1,0,1,0;1,0,1,0,1];  %This is for 3 actions  %T is the output and determines the output dimension of the nn.
% FIXME: I tried to use one hot vectors for classification but It doesn't seem to
% work because I'm getting bad outputs

% FIXME: initialize smarter (X,T doesn't really make sense)
net_Q.trainFcn = 'traingdm';          %Gradient Descent Backpropagation
net_Q.trainParam.epochs = 1;           %1 epoch becuase we are batch-training
net_Q.trainParam.lr = 0.0001;
net_Q.trainParam.mc = 0.99;
net_Q.trainParam.showWindow=true;    %show GUI for training results
net_Q.divideFcn='dividetrain';       %assign all examples to training (and not to validation/testing)
net_Q.performFcn='mse';             %mean square error performance function
%net_Q.performFcn='crossentropy';     %mean square error performance function
init(net_Q);                        %initialize net
net_Q=train(net_Q,X,T);


% x1=[1,2,3,4];
% net_Q(x1')
% view(net_Q)
% B=net_Q.b;
% IW=net_Q.IW;
% LW=net_Q.LW


%load values to network:
%this example is for a 4 input, 3 output,hidden layers: 10X10X40 
%% input layer:
% net1.IW

% ans =
% 
%   4ª1 cell array
% 
%     [5ª4 double]
%     []
%     []
%     []
% 
%     
% go to net1.IW{1}
% and then we can call it like
% net1.Iw{row,column}
% and assign the values we want

%%  middle and out layers
% net1.LW
% 
% net1.LW
% 
% ans =
% 
%   4ª4 cell array
% 
%               []               []               []    []
%     [5ª5 double]               []               []    []
%               []    [40ª5 double]               []    []
%               []               []    [3ª40 double]    []

%go to net1.LW{2,1} for example
%and then assign values like this: net1.LW{2,1}(1,1)=3

% for the out layer write net1.LW{4,3}
end

