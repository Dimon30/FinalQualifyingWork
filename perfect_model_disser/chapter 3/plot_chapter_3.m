%% Глава 2.5
clear all
close all
clc

% Установка параметров графика
w = 1600; 
h = 600; % Увеличим высоту для лучшего отображения
fs = 18; % Размер шрифта

% Запуск Simulink модели и получение результатов
model = 'quad_system_3'; % Замените на имя вашей модели
simOut = sim(model); % Запуск симуляции

% Извлечение данных времени и значений x, y, z из Simulink
time = simOut.get('tout'); % Время
X_real = squeeze(simOut.get('X').Data);   % Значения реального x
Y_real = squeeze(simOut.get('Y').Data);   % Значения реального y
Z_real = squeeze(simOut.get('Z').Data);   % Значения реального z
xstar1 = squeeze(simOut.get('x_h').Data);   % Значения заданного x
ystar1 = squeeze(simOut.get('y_h').Data);   % Значения заданного y
zstar1 = squeeze(simOut.get('z_h').Data);   % Значения заданного z

% Создание фигуры для первого примера с трехмерным графиком
figure;
set(gcf, 'Position', [250 250 w h])
set(gcf, 'Color', [1, 1, 1]); % Устанавливаем цвет фона фигуры на белый
set(gca, 'Color', [1, 1, 1]); % Устанавливаем цвет фона осей на белый
hold on;
grid on;

% Получаем текущие оси
ax = gca;
ax.GridLineStyle = '--'; % Прерывистая линия для основной сетки

% Построение желаемой траектории по ориентирам пунктирной красной линией
plot3(xstar1, ystar1, zstar1, '--r', 'LineWidth', 2, 'DisplayName', 'Заданная траектория');

% Построение реальной траектории из Simulink сплошной синей линией
plot3(X_real, Y_real, Z_real, 'LineWidth', 2, 'Color', [0.0078, 0.4470, 0.7410], 'DisplayName', 'Траектория квадрокоптера');

% Настройка осей и заголовка графика
xlabel('$x\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
ylabel('$y\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
zlabel('$z\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
%xlabel('x, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
%ylabel('у, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
%zlabel('z, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
legend('Location', 'NorthEast', 'FontSize', fs);
view(3); % Установка трехмерного вида
hold off;

% Создание графика X-Y
figure;
set(gcf, 'Position', [250 + w/2, 250, w/2, h/2]); % Исправлено на правильный синтаксис
hold on;
grid on;
ax = gca;
ax.GridLineStyle = '--'; % Прерывистая линия для основной сетки

plot(X_real, Y_real, 'LineWidth', 2, 'Color', [0.0078, 0.4470, 0.7410], 'DisplayName', 'Траектория квадрокоптера');
plot(xstar1, ystar1, '--r', 'LineWidth', 2, 'DisplayName', 'Заданная траектория');
legend('Location', 'NorthEast', 'FontSize', fs);
xlabel('x, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
ylabel('у, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
hold off;

% Создание графика X-Z
figure;
set(gcf, 'Position', [250 + w/2, 250 + h/2, w/2, h/2]); % Исправлено на правильный синтаксис
hold on;
grid on;
ax = gca;
ax.GridLineStyle = '--'; % Прерывистая линия для основной сетки

plot(X_real, Z_real, 'LineWidth', 2, 'Color', [0.0078, 0.4470, 0.7410], 'DisplayName', 'Траектория квадрокоптера');
plot(xstar1, zstar1, '--r', 'LineWidth', 2, 'DisplayName', 'Заданная траектория');
legend('Location', 'NorthEast', 'FontSize', fs);
xlabel('x, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
ylabel('z, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
hold off;

%% Errors

time = simOut.get('tout'); % Время
X_err = squeeze(simOut.get('X_err').Data);   % Значения X_err
Y_err = squeeze(simOut.get('Y_err').Data);   % Значения Y_err
Z_err = squeeze(simOut.get('Z_err').Data);   % Значения Z_err

% Создание фигуры для графиков ошибок
figure;
set(gcf, 'Position', [250 250 1600 600]); % Установка размера фигуры

% Настройка стиля линий сетки
ax = gca; % Получаем текущую ось
ax.XGrid = 'on'; % Включаем сетку по оси X
ax.YGrid = 'on'; % Включаем сетку по оси Y
ax.GridLineStyle = '--'; % Прерывистая линия

% Настройка осей
axis([min(time) max(time) -0.5 0.5]); % Установите диапазон осей по необходимости
%Построение графиков ошибок
plot(time, X_err, 'LineWidth', 3, 'Color', [0.4660 0.6740 0.1880], 'LineStyle', '--', 'DisplayName', '$x - x^*$');
hold on; % Держим текущий график для добавления новых линий
plot(time, Y_err, 'LineWidth', 3, 'Color', [0.9290 0.6940 0.1250], 'LineStyle', '-.', 'DisplayName', '$y - y^*$');
plot(time, Z_err, 'LineWidth', 3, 'Color', [0 0.4470 0.7410], 'LineStyle', '-', 'DisplayName', '$z - z^*$');

%Вычисление нормы ошибок
norm_error = sqrt(X_err.^2 + Y_err.^2 + Z_err.^2);

%Построение графика нормы ошибок
plot(time, norm_error, 'LineWidth', 3, 'Color', [1 0 0], 'LineStyle', '-', 'DisplayName', '$\sqrt{(x - x^*)^2 + (y - y^*)^2 + (z - z^*)^2}$');
% Подписи осей и легенда
xlabel('$t\,(\mathrm{c})$', 'FontSize', fs, 'Interpreter', 'latex');
%ylabel('$error\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
legend_output = legend('show'); % Используем show для отображения легенды
set(legend_output, 'Interpreter', 'latex', 'FontSize', fs);
grid on;
ax = gca;
ax.GridLineStyle = '--'; % Прерывистая линия для основной сетки
hold off; % Освобождаем график для будущих построений
