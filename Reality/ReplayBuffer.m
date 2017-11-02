classdef ReplayBuffer < handle
    %REPLAYBUFFER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
       buffer_size
       state_size
       action_size
       elemsInBuffer
       S
       A
       R
       S_tag
    end
    
    methods
        function obj = ReplayBuffer(buffer_size,state_size,action_size) %constructor
            obj.buffer_size = buffer_size;
            obj.state_size = state_size;
            obj.action_size = action_size;
            obj.elemsInBuffer=0;
            obj.S = [];
            obj.A = [];
            obj.R = [];
            obj.S_tag = [];
                
        end
        

      function store_transition(obj,s,a,r,s_tag) %transition looks like this [s,a,r,s_tag]
          %add transition to the ReplayBuffer
         obj.S = cat(1,obj.S,s');
         obj.A = cat(1,obj.A,a);
         obj.R = cat(1,obj.R,r);
         obj.S_tag = cat(1,obj.S_tag,s_tag');
         
         obj.elemsInBuffer = obj.elemsInBuffer+1;       %update num of elements in buffer
  
         if (obj.elemsInBuffer >obj.buffer_size)   %check if the buffer is full, and if so - delete from the beginning (FIFO)
             obj.S(1,:)=[];
             obj.A(1,:)=[];
             obj.R(1,:)=[];
             obj.S_tag(1,:)=[];
             
             obj.elemsInBuffer = obj.elemsInBuffer-1;   %update num of elements in buffer
         end
         
      end
      
      
       function [s_batch,a_batch,r_batch,s_tag_batch] = sample_mini_batch(obj,batch_size) %transition looks like this [s,a,r,s_tag]
           %check if there are enough transitions for batch extraction
           if (batch_size>obj.elemsInBuffer)    %not enough elements in the replay_buffer - just take all of the buffer for training
              s_batch = obj.S;
              a_batch = obj.A;
              r_batch = obj.R;
              s_tag_batch = obj.S_tag;
           
           else
               %enough transitions in replay buffer, sample them randomly for later training
              indices = randi([1,obj.elemsInBuffer], batch_size, 1);
              s_batch = obj.S(indices,:);
              a_batch = obj.A(indices,:);
              r_batch = obj.R(indices,:);
              s_tag_batch = obj.S_tag(indices,:);
           end
 
       end
      
       
    end
    
end

