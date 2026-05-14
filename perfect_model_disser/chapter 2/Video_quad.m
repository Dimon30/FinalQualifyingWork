%% Глава 2.5
clear all
close all
clc

% Запуск Simulink модели и получение результатов
model = 'quad_system_1'; % Замените на имя вашей модели
simOut = sim(model); % Запуск симуляции

% Извлечение данных времени и значений x, y, z из Simulink
time = simOut.get('tout'); % Время
X_real = simOut.get('X');   % Значения реального x
Y_real = simOut.get('Y');   % Значения реального y
Z_real = simOut.get('Z');   % Значения реального z

% Определение углов наклона квадрокоптера (можно оставить нулевыми)
roll_angles = zeros(size(X_real.data));   % Assuming no roll for simplicity
pitch_angles = zeros(size(Y_real.data));   % Assuming no pitch for simplicity

% Yaw angles can be calculated based on the trajectory direction.
yaw_angles = zeros(size(X_real.data)); 

for i = 2:length(X_real.data)
    dx = X_real.data(i) - X_real.data(i-1);
    dy = Y_real.data(i) - Y_real.data(i-1);
    yaw_angles(i) = atan2d(dy, dx); % Calculate yaw angle in degrees
end

% Animate using the drone_Animation function with real trajectory data
drone_Animation(X_real.data, Y_real.data, Z_real.data, roll_angles, pitch_angles, yaw_angles);