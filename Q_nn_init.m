function [ net_Q ] = Q_nn_init( Q_NET_SIZES,STATE_SIZE,ACTION_SIZE )
%Q_NN_INIT 
%   initialize Q_NET_SIZES sized NN for the DQN algorithm



%********************************define and init Q nn********************
net_Q=feedforwardnet(Q_NET_SIZES);

X=rand([STATE_SIZE,STATE_SIZE]);
T=rand([ACTION_SIZE,STATE_SIZE]);


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




