close all; clear; clc

%% enable camera for remote picture
%cam=webcam
%preview(cam)

%% define hyper-parameters for dqn
C = 50;                %number of cycles until we substitute (Q_target)<-(Q)
D = 200000;                %replay buffer size. each element in the buffer is (s,a,r,s')
Episodes = 5000;           %number of episodes for training
Loops = 300;             %number of loops (STEPS) in each episodes
BatchSize4Training=32;   %number of transitions to sample for batch training
Batches=200;             %number of batches training for each loop episode
epsilon_greedy=0.1;    %epsilon greedy policy for Q
Gamma = 0.999;          %discount factor for Q-learning
NumOfActions=5;         %number of actions in our discretized actions space
STRONG_ACTION_DISCRETIZATION = 10; %Voltage of 'strong action' discretization
ArrOfActions = linspace(-STRONG_ACTION_DISCRETIZATION,STRONG_ACTION_DISCRETIZATION,NumOfActions);  %array of actions
STATE_SIZE = 5;         % %state is a 5-tuple <x,x_dot,cos(theta),sin(theta),theta_dot>
curr_state=[0;0;1;0;0];   %state is a 5-tuple <x,x_dot,cos(theta),sin(theta),theta_dot>
R_tot = 0;              %R_tot is the cumulative reward of each episode
% R_tot_arr = [];         %array of all episodes' cumulative reward 
pos_goal = 0;           %our goal is to be close to where we started    %FIXME: WE CAN REMOVE THIS DEMAND
ang_goal = 180;         %our prime goal is to stabilize the penundulum in vertical state
state_goal = [pos_goal,0,cos(ang_goal*pi/180),sin(ang_goal*pi/180),0];
Q_NET_SIZES = [40,100];   % number of neurons in each layer of the 2-layer nn
DESIRED_LOOP_TIME = 0.06; %[Sec]    %deamand constant loop time for markov process
SAVE_RATE=50;               %save logs every SAVE_RATE times
%% constants for reward calculation
q = [5,0.1,7.5,0.1];
r = 0.01;
n = 0.1;
%% CONTROL PARAMETERS
POS_BOUNDARY = 70;      %the is not allowed to go further then POS_BOUNDARY by means of the controller in order to facilitate learning
% SWING_PERIOD = 10;      %swing the penundulum every SWING_PERIOD to give good initial conditions
%% initialize neural networks
% we use two nns - Q and Q_target
% 

if exist('logs/Q_real.mat', 'file') == 2 %check if a saved model exists
    load('logs/Q_real.mat');         % Q=Q_load;
    disp("Q-network loaded");
    load('logs/Q_target_real.mat');  % Q_target = Q_target_load;
    disp("Q_target-network loaded");
    load('logs/Reward_real.mat');
    disp("Reward array loaded");
else
    Q = Q_nn_init(Q_NET_SIZES,STATE_SIZE,NumOfActions);  
    Q_target = Q;
    R_tot_arr = [];         %array of all episodes' cumulative reward 
end
    


% Q.trainParam.lr = 0.0001;           %DECAY learning rate
%% initialize replay buffer

if exist('logs/rpBuffer_real.mat', 'file') == 2 %check if a replay buffer exists
    load('logs/rpBuffer_real.mat');
    disp("ReplayBuffer loaded");
else
    rpBuffer = ReplayBuffer(D,1,STATE_SIZE);%D is the buffer size; 1 is the action dimension(1,2,3,..); STATE_SIZE is the state dimension (curr_angle,prev_angle,curr_pos,prev_pos)
end

% rpBuffer = ReplayBuffer(D,1,5);%D is the buffer size; 1 is the action dimension(1,2,3,..); 4 is the state dimension (curr_angle,prev_angle,curr_pos,prev_pos)


%*****create sessions for reading/writing to the penundulum******
counterNBits = 32;
signedThreshold = 2^(counterNBits-1);

s = daq.createSession('ni');    %session for reading angle/position from encoders
s1 = daq.createSession('ni');   %session for writing analog outputs to the motor

ctrangle = addCounterInputChannel(s, 'Dev1', 1, 'Position'); %angle
ctrpos = addCounterInputChannel(s, 'Dev1', 0, 'Position'); %position
ctrangle.EncoderType = 'X4';
ctrpos.EncoderType = 'X4';

addAnalogOutputChannel(s1,'Dev1',1,'Voltage');
%*********end establishing connections via sessions*************


%% loop episode
for j=1:Episodes
%     Disp= ['Episode: ', num2str(j), ' starts'];
% 	disp(Disp);
%init "zero state" - (s_t,a_t)

%FIXME: decide how to do it

%     disp "start!";
%     pause(2)      %YOTAM: I DISABLED IT FOR TIME SAVING

    %*******************loop single*****************
    
    %********observe s once before the loop so we can observe during each loop only once.
%     1. observe s
    encang = inputSingleScan(s);
    signedData1 = encang(1);
    theta = mod(signedData1 * 360 /4096,360);
    signedData2 = encang(2);
    signedData2(signedData2 > signedThreshold) = signedData2(signedData2 > signedThreshold) - 2^counterNBits;
    x = signedData2*1.1*2*pi/2048 ;
    x_dot = (x-curr_state(1))/DESIRED_LOOP_TIME;
    theta_dot = ((theta*pi/180)-acos(curr_state(3)))/DESIRED_LOOP_TIME;
%     curr_state = [theta;curr_state(1);x;curr_state(3)];  
    curr_state = [x;x_dot;cos(theta*pi/180);sin(theta*pi/180);theta_dot];
    %******************finished observing once*********************
        
    for i=1:Loops
        tic             %use tic toc to measure time in order to get to DESIRED_LOOP_TIME


        %% 2. calc a = max_a(Q(s)) or rand(Q(s)) WITH THE RESTRICTIONS OF THE CONTROLLER
        % - this is e-greedy
        if (rand > 1-epsilon_greedy)   
        action2take = randi([1,NumOfActions],1,1);   %choose action randomlly from NumOfActions optional actions
        else
        [~,action2take] = max( Q(curr_state) ); %choose action from the Q_nn
        end
        
        %   CONTROLLER CHECKS CONDITIONS
        if abs( curr_state(1) ) >= POS_BOUNDARY           %curr_state(3) is the current position

            if curr_state(1) >= POS_BOUNDARY
                action2take = 1;
            else
                action2take = NumOfActions;
            end


        end

        %% 3. apply a WITH THE RESTRICTIONS OF THE CONTROLLER
        
        outputSingleScan(s1,ArrOfActions(action2take)); %apply action

        %% 4. observe s',r
        %here i should read data(state s') from the encoders
        encang = inputSingleScan(s);
        signedData1 = encang(1);
        theta = mod(signedData1 * 360 /4096,360);
        signedData2 = encang(2);
        signedData2(signedData2 > signedThreshold) = signedData2(signedData2 > signedThreshold) - 2^counterNBits;
        x = signedData2*1.1*2*pi/2048 ;
%         curr_state2 = [ang;curr_state(1);pos;curr_state(3)];
        x_dot = (x-curr_state(1))/DESIRED_LOOP_TIME;
        theta_dot = ((theta*pi/180)-acos(curr_state(3)))/DESIRED_LOOP_TIME;
        curr_state2 = [x;x_dot;cos(theta*pi/180);sin(theta*pi/180);theta_dot];
        %calc reward       

        curr_reward = -n*(q(1)*abs(x) + q(2)*abs(x_dot) - q(3)*(2-abs(1+cos(theta*pi/180))) + q(4)*abs(theta_dot) + r*abs(ArrOfActions(action2take)));
%         Disp= ['x: ', num2str(x),' x_dot: ', num2str(x_dot),' theta: ', num2str(theta),' theta_dot: ', num2str(theta_dot), ' ArrOfActions(action2take): ', num2str(ArrOfActions(action2take)), ' curr_reward: ', num2str(curr_reward)];
%         disp(Disp);

        R_tot = R_tot + curr_reward;

        %% 5. store (s,a,r,s') in the ReplayBuffer
        % here we construct a new state representation to be compatible
        % with Eyal's python representation
        %buffer state is - x,x_dot,theta,theta_dot, and theta in radians
        rpBuffer.store_transition(curr_state,action2take,curr_reward,curr_state2);
        curr_state = curr_state2;   %update current_state
        
        END_TIME = toc;
%         arr_END_TIME = [arr_END_TIME,END_TIME];
        if (DESIRED_LOOP_TIME - END_TIME) <0
            disp('problem')
        end
        pause( DESIRED_LOOP_TIME - END_TIME )
        
    end
    %*************end loop single*******************
    
    outputSingleScan(s1,0); %this is for resting the damn penundulum
    
    %********BRING PEN. TO THE RANDOM POSITION AFTER LOOP SINGLE**********
    %################################################################*
    
%     rand_pos = randi([-POS_BOUNDARY/2,POS_BOUNDARY/2],1,1);
    rand_pos = 0; %START WITH ZERO TO FACILITATE LEARNING


%     Disp = ['return to   initial location: ', num2str(rand_pos)];
%     disp(Disp);
    pos_b2c = x;
    while abs (pos_b2c - rand_pos) > 0.05
        encang_b2c = inputSingleScan(s);
        signedData1_b2c = encang_b2c(1);
        ang_b2c = mod(signedData1_b2c * 360 /4096,360);
        signedData2_b2c = encang_b2c(2);
        signedData2_b2c(signedData2_b2c > signedThreshold) = signedData2_b2c(signedData2_b2c > signedThreshold) - 2^counterNBits;
        pos_b2c = signedData2_b2c*1.1*2*pi/2048 ; 
%         disp(pos_b2c)
        outputSingleScan(s1,-sign(pos_b2c - rand_pos)*5); %apply action
    end
    
    outputSingleScan(s1,0); %this is for resting the damn penundulum



    
    %################################################################* 
    %************END BRINGING PEN. TO CENTER**************************
    
    outputSingleScan(s1,0); %this is for resting the damn penundulum

%     R_tot_arr = cat(1,R_tot_arr,R_tot); %save cumulative reward per episode to indicate learning
    R_tot_arr = [R_tot_arr , R_tot];
%   Disp= ['Reward: ', num2str(R_tot)];
% 	disp(Disp);
    % this is where we try to learn

    Disp= ['episode ', num2str(j),'/',num2str(Episodes),  '(',num2str(length(R_tot_arr)),')',', reward: ', num2str(R_tot)];
    disp(Disp);
%     disp('start learning');
    
    R_tot=0;
    for mini_batch = 1:Batches
%         tic
        %% 6. sample BatchSize4Training transitions (s,a,r,s') from D
        [batch_s,batch_a,batch_r,batch_s_tag] = rpBuffer.sample_mini_batch(BatchSize4Training);

        %% 7. gradient descent Q
        Y = Q(batch_s');    %s is in rows and the nn accepts s as a column vector
        Q_next = Q_target(batch_s_tag');
        Q_next_max = max(Q_next);

        for i=1:BatchSize4Training
           Y(batch_a(i),i) = batch_r(i) + Gamma*Q_next_max(i);
        end

        % train on estimated Q_next and rewards
        %sess.run(Q_train_step, {state_input:s_batch, y_estimate:Y})
        Q = train(Q,batch_s',Y);


    end
    
 %% 8. if number of loop singles=c: Q_target=Q

    if ( mod(j,C)==0)
        Q_target=Q;
        disp 'updated target network'
        
    end
    
    % decay learning rate every 1000 episodes
%     if ( mod(j,5000)==0)           
%         Q.trainParam.lr = Q.trainParam.lr/2;
%         disp(Q.trainParam.lr);
%     end
%     
% 	disp('stopped learning');
    
%     % check if its swing time!
%     if (mod (j,SWING_PERIOD) == 0)     %time to swing!
%         swing_me(s,s1);
%     end
% If you want to se guided DQN with swing, insert the above part to the
% learning loop and set a probability in which to apply the swing policy

    if ( mod(j,SAVE_RATE)==0)   %time to save logs
        save('logs/Q_real.mat','Q')
        save('logs/Q_target_real.mat','Q_target')
        save('logs/rpBuffer_real.mat','rpBuffer')
        save('logs/Reward_real.mat','R_tot_arr')
        disp("model parameters saved")
        
        if (mod(j,500)==0)  %create checkpoint because NI device can start being crazy
            Q_addr=['logs/Q_real',num2str(j),'.mat'];
            Q_target_addr=['logs/Q_target_real',num2str(j),'.mat'];
            rpBuffer_addr=['logs/rpBuffer_real',num2str(j),'.mat'];
            Reward_addr=['logs/Reward_real',num2str(j),'.mat'];

            save(Q_addr,'Q');
            save(Q_target_addr,'Q_target')
            save(rpBuffer_addr,'rpBuffer')
            save(Reward_addr,'R_tot_arr')
            disp("checkpoint added")
        end
        
    end
    
    if (mod (j,1000) == 0)
        save('logs/System_State_real.mat')
    end



end

save('System_State_real.mat')

    %********BRING PEN. TO ZERO STATE***************
    %################################################################*
    
    rand_pos = 0;
    Disp = ['return to CENTER: ', num2str(rand_pos)];
    disp(Disp);
    pos_b2c = x;
    while abs (pos_b2c - rand_pos) > 0.005
        encang_b2c = inputSingleScan(s);
        signedData1_b2c = encang_b2c(1);
        ang_b2c = mod(signedData1_b2c * 360 /4096,360);
        signedData2_b2c = encang_b2c(2);
        signedData2_b2c(signedData2_b2c > signedThreshold) = signedData2_b2c(signedData2_b2c > signedThreshold) - 2^counterNBits;
        pos_b2c = signedData2_b2c*1.1*2*pi/2048 ; 
%         disp(pos_b2c)
        outputSingleScan(s1,-sign(pos_b2c - rand_pos)*5); %apply action
    end

    %################################################################* 
    %************END BRINGING PEN. TO CENTER**************************
    

outputSingleScan(s1,0); %this is for resting the damn penundulum
