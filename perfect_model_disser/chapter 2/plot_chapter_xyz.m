%% Глава 2.5
clear all
close all
clc

% Установка параметров графика
w = 1600; 
h = 600; % Увеличим высоту для лучшего отображения
fs = 18; % Размер шрифта

% Запуск Simulink модели и получение результатов
model = 'quad_system_1'; % Замените на имя вашей модели
simOut = sim(model); % Запуск симуляции

% Извлечение данных времени и значений x, y, z из Simulink
time = simOut.get('tout'); % Время
X_real = simOut.get('X');   % Значения реального x
Y_real = simOut.get('Y');   % Значения реального y
Z_real = simOut.get('Z');   % Значения реального z
xstar1 = simOut.get('x_h');   % Значения реального x
ystar1 = simOut.get('y_h');   % Значения реального y
zstar1 = simOut.get('z_h');   % Значения реального z

% Создание фигуры для первого примера с трехмерным графиком
figure;
set(gcf, 'Position', [250 250 w h])
set(gcf, 'Color', [1, 1, 1]); % Устанавливаем цвет фона фигуры на белый
set(gca, 'Color', [1, 1, 1]); % Устанавливаем цвет фона осей на белый
hold on;
grid on;
% Получаем текущие оси
ax = gca;

% Устанавливаем стиль линий сетки
ax.GridLineStyle = '--'; % Прерывистая линия для основной сетки
% Построение желаемой траектории по ориентирам пунктирной красной линией
plot3(xstar1.data, ystar1.data, zstar1.data, '--r', 'LineWidth', 2, 'DisplayName', 'Заданная траектория');

% Построение реальной траектории из Simulink сплошной синей линией
plot3(X_real.data, Y_real.data, Z_real.data, ...
      'LineWidth', 2, 'Color', [0.0078, 0.4470, 0.7410], 'DisplayName', 'Траектория квадрокоптера');

% Настройка осей и заголовка графика
xlabel('x, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
ylabel('у, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
zlabel('z, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
legend('Location', 'NorthEast', 'FontSize', fs);
view(3); % Установка трехмерного вида
hold off;


% Создание графика X-Y
figure;
set(gcf, 'Position', [250 + w/2, 250 w/2 h/2]);
hold on;
grid on;
% Получаем текущие оси
ax = gca;

% Устанавливаем стиль линий сетки
ax.GridLineStyle = '--'; % Прерывистая линия для основной сетки
plot(X_real.data, Y_real.data, ...
     'LineWidth', 2, 'Color', [0.0078, 0.4470, 0.7410], 'DisplayName', 'Траектория квадрокоптера');
plot(xstar1.data, ystar1.data,'--r', 'LineWidth', 2,'DisplayName','Заданная траектория');

xlabel('x, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
ylabel('у, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
%title('График X-Y');
legend('Location','NorthEast', 'FontSize', fs);
hold off;

% Создание графика X-Z
figure;
set(gcf, 'Position', [250 + w/2, 250 + h/2 w/2 h/2]);
hold on;
grid on;
% Получаем текущие оси
ax = gca;

% Устанавливаем стиль линий сетки
ax.GridLineStyle = '--'; % Прерывистая линия для основной сетки
plot(X_real.data,Z_real.data,...
     'LineWidth',2,'Color',[0.0078,0.4470,0.7410],'DisplayName','Траектория квадрокоптера');
plot(xstar1.data,zstar1.data,'--r','LineWidth',2,'DisplayName','Заданная траектория');

xlabel('x, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
ylabel('z, м', 'FontSize', fs, 'Interpreter', 'none', 'FontAngle', 'italic');
%title('График X-Z');
legend('Location','NorthEast', 'FontSize', fs);
hold off;   

%% Errors
time = simOut.get('tout'); % Время
X_err = simOut.get('X_err');   % Значения X_err
Y_err = simOut.get('Y_err');   % Значения Y_err
Z_err = simOut.get('Z_err');   % Значения Z_err

%Создание фигуры для графиков ошибок
figure
set(gcf, 'Position', [250 250 1600 600]); % Установка размера фигуры

%Настройка стиля линий сетки
ax = gca; % Получаем текущую ось
ax.XGrid = 'on'; % Включаем сетку по оси X
ax.YGrid = 'on'; % Включаем сетку по оси Y

%Установка стиля линий сетки
ax.GridLineStyle = '--'; % Прерывистая линия
%Настройка осей
axis([min(time) max(time) -0.5 0.5]); % Установите диапазон осей по необходимости
%Построение графиков ошибок
plot(time, X_err.data, 'LineWidth', 3, 'Color', [0.4660 0.6740 0.1880], 'LineStyle', '--', 'DisplayName', '$x - x^*$');
hold on; % Держим текущий график для добавления новых линий
plot(time, Y_err.data, 'LineWidth', 3, 'Color', [0.9290 0.6940 0.1250], 'LineStyle', '-.', 'DisplayName', '$y - y^*$');
plot(time, Z_err.data, 'LineWidth', 3, 'Color', [0 0.4470 0.7410], 'LineStyle', '-', 'DisplayName', '$z - z^*$');

%Вычисление нормы ошибок
norm_error = sqrt(X_err.data.^2 + Y_err.data.^2 + Z_err.data.^2);

%Построение графика нормы ошибок
plot(time, norm_error, 'LineWidth', 3, 'Color', [1 0 0], 'LineStyle', '-', 'DisplayName', '$\sqrt{(x - x^*)^2 + (y - y^*)^2 + (z - z^*)^2}$');

%Подписи осей и легенда
xlabel('$t\,(\mathrm{c})$', 'FontSize', fs, 'Interpreter', 'latex');
ylim([-10 10]); % Увеличиваем диапазон по оси Y для лучшего отображения легенды
legend_output = legend('show'); % Используем show для отображения легенды
set(legend_output, 'Interpreter', 'latex', 'FontSize', fs);
grid on;
% %Получаем текущие оси
ax = gca;
% %Устанавливаем стиль линий сетки
ax.GridLineStyle = '--'; % Прерывистая линия для основной сетки
hold off; % Освобождаем график для будущих построений
