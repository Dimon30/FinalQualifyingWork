clear all
close all
clc

% Установка параметров графика
w = 1600; 
h = 400; % Увеличиваем высоту для двух графиков
fs = 18; % Размер шрифта

% Запуск Simulink модели и получение результатов
model = 'quadrotor_2021b_06112024_Line'; % Замените на имя вашей модели
simOut = sim(model); % Запуск симуляции

% Извлечение данных времени и значений phi, psi, theta
time = simOut.get('tout'); % Время
phi = simOut.get('phi');   % Значения phi
psi = simOut.get('psi');   % Значения psi
theta = simOut.get('theta'); % Значения theta

% Извлечение одномерных значений с помощью squeeze
phi_values = squeeze(phi.Data);
psi_values = squeeze(psi.Data);
theta_values = squeeze(theta.Data);

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
axis([min(time) max(time) -3 3]); % Установите диапазон осей по необходимости

% Построение графиков
plot(time, psi_values, 'LineWidth', 3, 'Color', [0.4660 0.6740 0.1880], 'LineStyle', '--');
plot(time, theta_values, 'LineWidth', 3, 'Color', [0.9290 0.6940 0.1250], 'LineStyle', '-.');
plot(time, phi_values, 'LineWidth', 3, 'Color', [0 0.4470 0.7410], 'LineStyle', '-');

% Подписи осей и легенда
xlabel('$t\,(\mathrm{c})$', 'FontSize', fs, 'Interpreter', 'latex');
ylabel('$\psi,\,\theta,\,\phi\,(\mathrm{rad})$', 'FontSize', fs, 'Interpreter', 'latex');
legend_output = legend('$\psi$', '$\theta$','$\phi$', 'Location', 'NorthEast');
set(legend_output, 'Interpreter', 'latex', 'FontSize', fs);

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
axis([min(time) max(time) 0 1.2]); % Установите диапазон осей по необходимости

% Вычисление cos(psi) * cos(theta)
cos_values = cos(psi_values) .* cos(theta_values);

% Построение графика
plot(time, cos_values, 'LineWidth', 3, 'Color', [0 0.4470 0.7410], 'LineStyle', '-');

% Подписи осей и легенда для второго графика
xlabel('$t\,(\mathrm{c})$', 'FontSize', fs, 'Interpreter', 'latex');
ylabel('$\cos(\psi) \cdot \cos(\theta)$', 'FontSize', fs, 'Interpreter', 'latex');
alpha = 0.9;
y_line_value = 1 / (1 + alpha);
yline(y_line_value, 'r--', 'LineWidth', 2); % Добавление горизонтальной линии

% Обновленная легенда для второго графика с правильными подписями
legend_output = legend({'$\frac{1}{1+\alpha}$', '$\cos(\psi) \cdot \cos(\theta)$'}, ...
                       'Location', 'NorthEast');
set(legend_output, 'Interpreter', 'latex', 'FontSize', fs); 

hold off;
