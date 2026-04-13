%% Глава 2.5
clear all
close all
clc

% Установка параметров графика
w = 1600; 
h = 600; % Увеличим высоту для лучшего отображения
fs = 18; % Размер шрифта

% Запуск Simulink модели и получение результатов
model = 'quadrotor_2021b_03112024'; % Замените на имя вашей модели
simOut = sim(model); % Запуск симуляции

% Предполагаем, что в simOut есть необходимые данные
% Извлечение данных из simOut
xstar1 = simOut.x_h; % Замените на соответствующие поля вашей модели
ystar1 = simOut.y_h; % Замените на соответствующие поля вашей модели
zstar1 = simOut.z_h; % Замените на соответствующие поля вашей модели

X = simOut.X; % Данные по оси X
Y = simOut.Y; % Данные по оси Y
Z = simOut.Z; % Данные по оси Z

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
plot3(xstar1.signals.values, ystar1.signals.values, zstar1.signals.values, '--r', 'LineWidth', 2, 'DisplayName', 'Заданная траектория');

% Построение реальной траектории из Simulink сплошной синей линией
plot3(X.time, X.signals.values, Y.time, Y.signals.values, Z.time, Z.signals.values, ...
      'LineWidth', 2, 'Color', [0.0078, 0.4470, 0.7410], 'DisplayName', 'Траектория квадрокоптера');

% Настройка осей и заголовка графика
xlabel('$x\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
ylabel('$y\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
zlabel('$z\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
legend('Location', 'NorthEast', 'FontSize', fs);
view(3); % Установка трехмерного вида
hold off;

% % Создание графика X-Y
% figure;
% set(gcf, 'Position', [250 + w/2, 250 w/2 h/2]);
% hold on;
% grid on;
% % Получаем текущие оси
% ax = gca;
% 
% % Устанавливаем стиль линий сетки
% ax.GridLineStyle = '--'; % Прерывистая линия для основной сетки
% plot(X_real.data, Y_real.data, ...
%      'LineWidth', 2, 'Color', [0.0078, 0.4470, 0.7410], 'DisplayName', 'Траектория квадрокоптера');
% plot(xstar1, ystar1,'--r', 'LineWidth', 2,'DisplayName','Заданная траектория');
% 
% xlabel('$x\,(\mathrm{m})$', 'FontSize', fs,'Interpreter','latex');
% ylabel('$y\,(\mathrm{m})$', 'FontSize', fs,'Interpreter','latex');
% %title('График X-Y');
% %legend('Location','NorthEast');
% hold off;

% % Создание графика X-Z
% figure;
% set(gcf, 'Position', [250 + w/2, 250 + h/2 w/2 h/2]);
% hold on;
% grid on;
% % Получаем текущие оси
% ax = gca;
% 
% % Устанавливаем стиль линий сетки
% ax.GridLineStyle = '--'; % Прерывистая линия для основной сетки
% plot(X_real.data,Z_real.data,...
%      'LineWidth',2,'Color',[0.0078,0.4470,0.7410],'DisplayName','Траектория квадрокоптера');
% plot(xstar1,zstar1,'--r','LineWidth',2,'DisplayName','Заданная траектория');
% 
% xlabel('$x\,(\mathrm{m})$','FontSize',fs,'Interpreter','latex');
% ylabel('$z\,(\mathrm{m})$','FontSize',fs,'Interpreter','latex');
% %title('График X-Z');
% %legend('Location','NorthEast');
% hold off;   

% %% Errors
% %Извлечение данных времени и значений phi, psi, theta
% time = simOut.get('tout'); % Время
% X_err = simOut.get('X_err');   % Значения X_err
% Y_err = simOut.get('Y_err');   % Значения Y_err
% Z_err = simOut.get('Z_err');   % Значения Z_err
% 
% %Создание фигуры для графиков ошибок
% figure
% set(gcf, 'Position', [250 250 1600 600]); % Установка размера фигуры
% 
% %Настройка стиля линий сетки
% ax = gca; % Получаем текущую ось
% ax.XGrid = 'on'; % Включаем сетку по оси X
% ax.YGrid = 'on'; % Включаем сетку по оси Y
% 
% %Установка стиля линий сетки
% ax.GridLineStyle = '--'; % Прерывистая линия
% 
% %Настройка осей
% axis([min(time) max(time) -0.5 0.5]); % Установите диапазон осей по необходимости
% 
% %Построение графиков ошибок
% plot(time, X_err.data, 'LineWidth', 3, 'Color', [0.4660 0.6740 0.1880], 'LineStyle', '--', 'DisplayName', '$X_{err}$');
% hold on; % Держим текущий график для добавления новых линий
% plot(time, Y_err.data, 'LineWidth', 3, 'Color', [0.9290 0.6940 0.1250], 'LineStyle', '-.', 'DisplayName', '$Y_{err}$');
% plot(time, Z_err.data, 'LineWidth', 3, 'Color', [0 0.4470 0.7410], 'LineStyle', '-', 'DisplayName', '$Z_{err}$');
% 
% %Вычисление нормы ошибок
% norm_error = sqrt(X_err.data.^2 + Y_err.data.^2 + Z_err.data.^2);
% 
% %Построение графика нормы ошибок
% plot(time, norm_error, 'LineWidth', 3, 'Color', [1 0 0], 'LineStyle', '-', 'DisplayName', '$\sqrt{X_{err}^2 + Y_{err}^2 + Z_{err}^2}$');
% 
% %Подписи осей и легенда
% xlabel('$t\,(\mathrm{s})$', 'FontSize', fs, 'Interpreter', 'latex');
% ylabel('$error\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
% legend_output = legend('show'); % Используем show для отображения легенды
% set(legend_output, 'Interpreter', 'latex', 'FontSize', fs);
% grid on;
% %Получаем текущие оси
% ax = gca;
% 
% %Устанавливаем стиль линий сетки
% ax.GridLineStyle = '--'; % Прерывистая линия для основной сетки
% hold off; % Освобождаем график для будущих построений
