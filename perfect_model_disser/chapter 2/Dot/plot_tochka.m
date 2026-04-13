clear all
close all
clc

% Установка параметров графика
w = 1600; 
h = 400; % Увеличиваем высоту для двух графиков
fs = 18; % Размер шрифта

% Запуск Simulink модели и получение результатов
model = 'quad_system_1'; % Замените на имя вашей модели
simOut = sim(model); % Запуск симуляции

% Извлечение данных времени и значений phi, psi, theta
time = simOut.get('tout'); % Время
phi = simOut.get('X');   % Значения phi
psi = simOut.get('Y');   % Значения psi
theta = simOut.get('Z'); % Значения theta
X_err = simOut.get('X_err');
Y_err = simOut.get('Y_err');
Z_err = simOut.get('Z_err');
% Создание фигуры
figure(1)
set(gcf, 'Position', [250 250 w h])

% Первый график: psi и theta
subplot(1, 2, 1); % Два графика в одном столбце
hold on
grid on
% Настройка стиля линий сетки
ax = gca; % Получаем текущую ось
ax.XGrid = 'on'; % Включаем сетку по оси X
ax.YGrid = 'on'; % Включаем сетку по оси Y

% Установка стиля линий сетки
ax.GridLineStyle = '--'; % Прерывистая линия
% Настройка осей
axis([min(time) max(time) 0 2.5]); % Установите диапазон осей по необходимости

% Построение графиков
plot(time, psi.data, 'LineWidth', 3, 'Color', [0.4660 0.6740 0.1880], 'LineStyle', '--');
plot(time, theta.data, 'LineWidth', 3, 'Color', [0.9290 0.6940 0.1250], 'LineStyle', '-.');
plot(time, phi.data, 'LineWidth', 3, 'Color', [0 0.4470 0.7410], 'LineStyle', '-');

% Подписи осей и легенда
xlabel('$t\,(\mathrm{c})$', 'FontSize', fs, 'Interpreter', 'latex');
%ylabel('$x,\, y,\, z\ $', 'FontSize', fs, 'Interpreter', 'latex');
fs = 18; % Установите размер шрифта
fs = 18; % Установите нужный размер шрифт
legend_output = legend('x, м', 'y, м', 'z, м', 'Location', 'NorthEast');
set(legend_output, 'Interpreter', 'none', 'FontSize', fs);

hold off;

% Второй график: cos(psi) * cos(theta)
subplot(1, 2, 2);
hold on
grid on

% Настройка стиля линий сетки
ax = gca; % Получаем текущую ось
ax.XGrid = 'on'; % Включаем сетку по оси X
ax.YGrid = 'on'; % Включаем сетку по оси Y

% Установка стиля линий сетки
ax.GridLineStyle = '--'; % Прерывистая линия
axis([min(time) max(time) -2.5 0.5]); % Установите диапазон осей по необходимости
y_line_value = 1 / (1 + alpha);

% Построение графиков
plot(time, X_err.data, 'LineWidth', 3, 'Color', [0.4660 0.6740 0.1880], 'LineStyle', '--', 'DisplayName', '$x - x^*$');
plot(time, Y_err.data, 'LineWidth', 3, 'Color', [0.9290 0.6940 0.1250], 'LineStyle', '-.', 'DisplayName', '$y - y^*$');
plot(time, Z_err.data, 'LineWidth', 3, 'Color', [0 0.4470 0.7410], 'LineStyle', '-', 'DisplayName', '$z - z^*$');

% Подписи осей и легенда
xlabel('$t\,(\mathrm{c})$', 'FontSize', fs, 'Interpreter', 'latex');
legend_output = legend('$x - x^*$', '$y - y^*$', '$z - z^*$', 'Location', 'NorthEast');
set(legend_output, 'Interpreter', 'latex', 'FontSize', fs);
hold off;