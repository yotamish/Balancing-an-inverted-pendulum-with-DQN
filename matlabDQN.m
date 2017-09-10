close all; clear; clc

%% define hyper-parameters for dqn
c = 30;                %number of cycles until we substitute (Q_target)<-(Q)
D = 100000;                %replay buffer size. each element in the buffer is (s,a,r,s')
Episodes = 5000;           %number of episodes for training
Loops = 200;             %number of loops (STEPS) in each episodes
BatchSize4Training=32;   %number of transitions to sample for batch training
Batches=50;             %number of batches training for each loop episode
epsilon_greedy=0.01;    %epsilon greedy policy for Q
Gamma = 0.999;          %discount factor for Q-learning
NumOfActions=3;         %number of actions in our discretized actions space
STRONG_ACTION_DISCRETIZATION = 7; %Voltage of 'strong action' discretization
% ArrOfActions = [-STRONG_ACTION_DISCRETIZATION,-STRONG_ACTION_DISCRETIZATION/2,0,STRONG_ACTION_DISCRETIZATION/2,STRONG_ACTION_DISCRETIZATION];  %array of actions
ArrOfActions = [-STRONG_ACTION_DISCRETIZATION,0,STRONG_ACTION_DISCRETIZATION];  %array of actions
curr_state=[0;0;0;0];   %state is (angle1,angle0,position1,position0)
R_tot = 0;              %R_tot is the cumulative reward of each episode
R_tot_arr = [];         %array of all episodes' cumulative reward 
pos_goal = 0;           %our goal is to be close to where we started    %FIXME: WE CAN REMOVE THIS DEMAND
ang_goal = 180;         %our prime goal is to stabilize the penundulum in vertical state
state_goal = [ang_goal;ang_goal;pos_goal;pos_goal];
Weight_mat = 1e-3*[10,0,0,0;0,10,0,0;0,0,1,0;0,0,0,1];   %weight matrice for the satate part of the reward
ACTION_TIME = 0.05;      %time period for action (seconds)
WeightsPerLayer = 80;   % number of neurons in each layer of the 2-layer nn
DESIRED_LOOP_TIME = 0.035; %[Sec]    %deamand constant loop time for markov process
%% CONTROL PARAMETERS
POS_BOUNDARY = 30;      %the is not allowed to go further then POS_BOUNDARY by means of the controller in order to facilitate learning
SWING_PERIOD = 7;      %swing the penundulum every SWING_PERIOD to give good initial conditions
POS_BOUNDARY_SWING = 3; %location barrier for swinging
%% initialize neural networks
% we use two nns - Q and Q_target
% 
% Q = Q_nn_init(WeightsPerLayer);  
% Q_target = Q;

%*****************LOADING Q,Q_TARGET FROM FILE**********************
load('Q_load_new_try.mat');         % Q=Q_load;
disp("Q loaded");
load('Q_target_load_new_try.mat');  % Q_target = Q_target_load;
disp("Q_target_cos_reward loaded");

% Q.trainParam.showWindow=false;    %show GUI for training results
% Q_target.trainParam.showWindow=false;    %show GUI for training results
% Q.trainParam.lr=0.0001;
% %*****************FINISH LOADING Q,Q_TARGET FROM FILE***************
%% initialize replay buffer

% rpBuffer = ReplayBuffer(D,1,4);%D is the buffer size; 1 is the action dimension(1,2,3,..); 4 is the state dimension (curr_angle,prev_angle,curr_pos,prev_pos)

%***********LOAD BUFFER FROM FILE***************
load('rpBuffer_new_try.mat');
disp("Replay Buffer_cos_reward Loaded");
%***********LOAD BUFFER FROM FILE***************

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
    Disp= ['Episode: ', num2str(j), ' starts'];
	disp(Disp);


    %*******************loop single*****************
    
    %********observe s once before the loop so we can observe during each loop only once.
%     1. observe s
    encang = inputSingleScan(s);
    signedData1 = encang(1);
    ang = mod(signedData1 * 360 /4096,360);
    signedData2 = encang(2);
    signedData2(signedData2 > signedThreshold) = signedData2(signedData2 > signedThreshold) - 2^counterNBits;
    pos = signedData2*1.1*2*pi/2048 ;
    curr_state = [ang;curr_state(1);pos;curr_state(3)];

    %******************finished observing once*********************
    if (mod(j,100) == 0)
        SWING_PERIOD = SWING_PERIOD+2;
    end
    %update SWING_PERIOD for gDQN (the determination
    %'guided/not-guided' is deterministic for now    
    for i=1:Loops
        tic             %use tic toc to measure time in order to get to DESIRED_LOOP_TIME

        %update SWING_PERIOD for gDQN (the determination
        %'guided/not-guided' is deterministic for now
        
        
        
        %% 2. calc a = max_a(Q(s)) or rand(Q(s)) WITH THE RESTRICTIONS OF THE CONTROLLER
        % - this is e-greedy
        if (rand > 1-epsilon_greedy)   
        action2take = randi([1,NumOfActions],1,1);   %choose action randomlly from NumOfActions optional actions
        else
        [~,action2take] = max( Q(curr_state) ); %choose action from the Q_nn
        end
        %   CONTROLLER CHECKS CONDITIONS
        if abs( curr_state(3) ) >= POS_BOUNDARY           %curr_state(3) is the current position
            action2take = (NumOfActions+1)/2;    %do nothing but with better programming
        else   %do a sort of guided DQN
            if (mod (j,SWING_PERIOD) == 0)  %time for guiding the DQN
                 if abs( curr_state(3) ) >= POS_BOUNDARY_SWING           %curr_state(3) is the current position
        %             action2take=(NumOfActions+1)/2;         % IF WE TRY TO PASS THE ALLOWED SUBSPACE, WE DO NOTHING
                    if curr_state(3) > 0
                        action2take = 1;
                    else
                        action2take=3;
                    end
                else            %if we're not at a boundary, try to swing!
        %             if ( curr_state(3) - curr_state(4) ) >= 0 %WE'RE GOING RIGHT! curr_state(3) is the current position, curr_state(4) is the previous position
        %                 action2take = 3                        %keep going right!
        %             else                                        %WE'RE GOING LEFT!
        %                 action2take = 1   

                    if ( curr_state(3) - curr_state(4) ) >= 0 %pendulum is going right
                        action2take = 3 ;                       %keep going right!
                    else                                        %WE'RE GOING LEFT!
                        action2take = 1;        

                    end
                 end
            
            end
%             end

        end

        %% 3. apply a WITH THE RESTRICTIONS OF THE CONTROLLER
        
        outputSingleScan(s1,ArrOfActions(action2take)); %apply action

        %% 4. observe s',r
        %here i should read data(state s') from the encoders
        encang = inputSingleScan(s);
        signedData1 = encang(1);
        ang = mod(signedData1 * 360 /4096,360);
        signedData2 = encang(2);
        signedData2(signedData2 > signedThreshold) = signedData2(signedData2 > signedThreshold) - 2^counterNBits;
        pos = signedData2*1.1*2*pi/2048 ;
        curr_state2 = [ang;curr_state(1);pos;curr_state(3)];

        %calc reward
        %funtion
%         curr_reward = -( (norm(curr_state2 - state_goal))^2 + action2take^2 ) ;  %no weight matrices
%         curr_reward = -( (curr_state2 - state_goal)'*Weight_mat*(curr_state2 - state_goal));% + (ArrOfActions(action2take)) ^ 2);   %with weight matrice 
        curr_reward = -cos(curr_state2(1)*pi/180);
        %try to add a 'win bonus'
        
%         diff_state = curr_state2-state_goal;
%         curr_reward = -1e-3*( (diff_state(1))' * (diff_state(1)) + (diff_state(2))' * (diff_state(2) ) );   %ONLY ANGLES!!!

        R_tot = R_tot + curr_reward;

        %% 5. store (s,a,r,s') in the ReplayBuffer
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
    
    %********BRING PEN. TO  RANDOM POSITION AFTER LOOP SINGLE**********
    %################################################################*
    
%     rand_pos = randi([-POS_BOUNDARY/2,POS_BOUNDARY/2],1,1);
    rand_pos = 0; %START WITH ZERO TO FACILITATE LEARNING


%     Disp = ['return to   initial location: ', num2str(rand_pos)];
%     disp(Disp);
    pos_b2c = pos;
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
    R_tot=0;
    % this is where we try to learn

    disp('start learning');
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

    if ( mod(j,c)==0)
        Q_target=Q;
        disp 'updated target network'
        
    end
    
    % decay learning rate every 1000 episodes
%     if ( mod(j,1000)==0)           
%         Q.trainParam.lr = Q.trainParam.lr/2;
%         disp(Q.trainParam.lr);
%     end
%     
	disp('stopped learning');
    
%     % check if its swing time!
%     if (mod (j,SWING_PERIOD) == 0)     %time to swing!
%         swing_me(s,s1);
%     end
 
    save('Q_load_new_try.mat','Q')
    save('Q_target_load_new_try.mat','Q_target')
    save('rpBuffer_new_try.mat','rpBuffer')
    save('Reward_new_try.mat','R_tot_arr')
    
    if (mod (j,1000) == 0)
        save('System_new_try.mat')
    end



end

save('System_all_vars_cos_reward_0.035.mat')

    %********BRING PEN. TO ZERO STATE***************
    %################################################################*
    
    rand_pos = 0;
    Disp = ['return to CENTER: ', num2str(rand_pos)];
    disp(Disp);
    pos_b2c = pos;
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
