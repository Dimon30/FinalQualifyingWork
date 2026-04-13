close all
clear all
clc
 
 %% 1. define the motion coordinates 
 % roll , pitch and yaw input in degree 
    t    = 0:0.03:10;   % simulation time for 10 second
    z     = t/3;        % z in meter 
    frq   = 0.25;
    frq_rp = 4*frq;
    y     = sin(2*pi*frq*t);   
    x     = cos(2*pi*frq*t);
    yaw   = (2*pi*frq*t)*(180/pi); % yaw in degree (full rotation in simulation period)
    roll  = 3*sin(2*pi*frq_rp*t);  % 3 degree sinusoidal roll input for test
    pitch = 3*cos(2*pi*frq_rp*t);  % 3 degree sinusoidal pitch input for test

 %% 6. animate by using the function makehgtform
 % Function for ANimation of QuadCopter
  drone_Animation(x,y,z,roll,pitch,yaw)
 
 
%  %% step5: Save the movie
% myWriter = VideoWriter('drone_animation', 'Motion JPEG AVI');
% myWriter = VideoWriter('drone_animation1', 'MPEG-4');
% myWriter.Quality = 100;
% myWritter.FrameRate = 120;
% 
% % Open the VideoWriter object, write the movie, and class the file
% open(myWriter);
% writeVideo(myWriter, movieVector);
% close(myWriter); 