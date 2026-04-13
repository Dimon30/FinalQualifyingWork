function drone_Animation(x, y, z, roll, pitch, yaw, xstar1, ystar1, zstar1)

    % Параметры
    D2R = pi / 180; % Градусы в радианы
    b = 0.6;        % Длина квадрата, покрываемого квадрокоптером (м)
    a = b / 3;      % Длина малого квадрата основания
    H = 0.06;       % Высота корпуса квадрокоптера
    H_m = H + H / 2; % Высота моторного цилиндра
    r_p = b / 4;    % Радиус пропеллеров

    % Создание окна и построение траектории
    figure('WindowState','fullscreen');
    hold on;
    grid on;
%    opengl hardware; % Включение аппаратного ускорения

    % Построение заданной траектории
    plot3(xstar1, ystar1, zstar1, '--r', 'LineWidth', 2, 'DisplayName', 'Заданная траектория');

    % Установка пределов осей и подписей
    fs = 18; % Размер шрифта
    xlabel('$x\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
    ylabel('$y\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
    zlabel('$z\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
    view(3); % 3D-вид

    %% Design Different parts
    % Построение базового квадрата
    ro = 45 * D2R; % Угол поворота основания
    Ri = [cos(ro) -sin(ro) 0; sin(ro) cos(ro) 0; 0 0 1];

    base_co = [-a/2  a/2  a/2 -a/2; -a/2 -a/2  a/2  a/2; 0 0 0 0];
    base = Ri * base_co;

    drone(1) = patch(base(1,:), base(2,:), base(3,:), 'r'); % Основание квадрокоптера
    drone(2) = patch(base(1,:), base(2,:), base(3,:) + H, 'r'); % Верх квадрокоптера
    alpha(drone(1:2), 0.7);

    % Построение перпендикулярных опор (ног)
    [xcylinder, ycylinder, zcylinder] = cylinder([H / 2 H / 2]);

    drone(3) = surface(b * zcylinder - b / 2, ycylinder, xcylinder + H / 2, 'facecolor', 'b');
    drone(4) = surface(ycylinder, b * zcylinder - b / 2, xcylinder + H / 2, 'facecolor', 'b'); 
    alpha(drone(3:4), 0.6);

    % Построение моторных цилиндров
    drone(5) = surface(xcylinder + b / 2, ycylinder, H_m * zcylinder + H / 2, 'facecolor', 'r');
    drone(6) = surface(xcylinder - b / 2, ycylinder, H_m * zcylinder + H / 2, 'facecolor', 'r');
    drone(7) = surface(xcylinder, ycylinder + b / 2, H_m * zcylinder + H / 2, 'facecolor', 'r');
    drone(8) = surface(xcylinder, ycylinder - b / 2, H_m * zcylinder + H / 2, 'facecolor', 'r');
    alpha(drone(5:8), 0.7);

    % Построение пропеллеров
    to = linspace(0, 2 * pi);
    xp = r_p * cos(to);
    yp = r_p * sin(to);

    drone(9)  = patch(xp + b / 2, yp, zeros(size(xp)) + (H_m + H / 2), 'c', 'LineWidth', 0.5);
    drone(10) = patch(xp - b / 2, yp, zeros(size(xp)) + (H_m + H / 2), 'c', 'LineWidth', 0.5);
    drone(11) = patch(xp, yp + b / 2, zeros(size(xp)) + (H_m + H / 2), 'p', 'LineWidth', 0.5);
    drone(12) = patch(xp, yp - b / 2, zeros(size(xp)) + (H_m + H / 2), 'p', 'LineWidth', 0.5);

    alpha(drone(9:12), 0.3);

    % Создание группы объектов и трансформаций
    combinedobject = hgtransform('parent', gca);
    set(drone, 'parent', combinedobject);

    %% Анимация
    for i = 1:length(x)
        plot3(x(1:i), y(1:i), z(1:i), 'b:', 'LineWidth', 1.5);

        translation = makehgtform('translate', [x(i), y(i), z(i)]);
        rotationMatrix = makehgtform('xrotate', roll(i) * D2R) * ...
                         makehgtform('yrotate', pitch(i) * D2R) * ...
                         makehgtform('zrotate', yaw(i) * D2R);

        set(combinedobject, 'matrix', translation * rotationMatrix);

        if mod(i, 10) == 0
            drawnow limitrate;
        end
        pause(0.000001);
    end

end
