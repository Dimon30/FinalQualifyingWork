
%% Глава 2.5
clear all
close all
clc

% Установка параметров графика
w = 1600; 
h = 600; % Увеличим высоту для лучшего отображения
fs = 18; % Размер шрифта

% Запуск Simulink модели и получение результатов
model = 'quadrotor_2021b_06112024_Spring'; % Замените на имя вашей модели
simOut = sim(model); % Запуск симуляции

% Извлечение данных времени и значений x, y, z из Simulink
time = simOut.get('tout'); % Время
X_real = squeeze(simOut.get('X').Data);   % Значения реального x
Y_real = squeeze(simOut.get('Y').Data);   % Значения реального y
Z_real = squeeze(simOut.get('Z').Data);   % Значения реального z
xstar1 = squeeze(simOut.get('x_h').Data);   % Значения заданного x
ystar1 = squeeze(simOut.get('y_h').Data);   % Значения заданного y
zstar1 = squeeze(simOut.get('z_h').Data);   % Значения заданного z
% Определение углов наклона квадрокоптера (можно оставить нулевыми)
roll_angles = zeros(size(X_real));   
pitch_angles = zeros(size(Y_real));   
yaw_angles = zeros(size(X_real)); 

% for i = 2:length(X_real)
%     dx = X_real(i) - X_real(i-1);
%     dy = Y_real(i) - Y_real(i-1);
%     yaw_angles(i) = atan2d(dy, dx); % Угол поворота в градусах
% end

% Анимация с записью видео, включая заданную траекторию
drone_Animation(X_real, Y_real, Z_real, roll_angles, pitch_angles, yaw_angles, xstar1, ystar1, zstar1);