function animation = drone_Animation(x, y, z, roll, pitch, yaw, xstar1, ystar1, zstar1)

% This Animation code is for QuadCopter.

%% Define design parameters
D2R = pi/180;
R2D = 180/pi;
b   = 0.6;   % the length of total square cover by whole body of quadcopter in meter
a   = b/3;   % the legth of small square base of quadcopter(b/4)
H   = 0.06;  % height of drone in Z direction (4cm)
H_m = H + H/2; % height of motor in z direction (5 cm)
r_p = b/4;   % radius of propeller

% % Create Figure and Plot Desired Trajectory
% figure;
% hold on;
% grid on;
% 
% % Построение заданной траектории пунктирной красной линией
% plot3(xstar1, ystar1, zstar1, '--r', 'LineWidth', 2, 'DisplayName', 'Заданная траектория');

%% Define waypoints A, B, C, D for fixed figure
A = [5, 5, 5];
B = [10, 10, 5];
C = [20, 5, 10];
D = [30, 20, 20];

% Create Figure and Plot Desired Trajectory
figure;
hold on;
grid on;

% Plot desired trajectory with dashed red lines
plot3([A(1), B(1)], [A(2), B(2)], [A(3), B(3)], '--r', 'LineWidth', 2);
plot3([B(1), C(1)], [B(2), C(2)], [B(3), C(3)], '--r', 'LineWidth', 2);
plot3([C(1), D(1)], [C(2), D(2)], [C(3), D(3)], '--r', 'LineWidth', 2);

%% Set axis limits and labels
fs = 18; % Размер шрифта
xlabel('$x\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
ylabel('$y\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
zlabel('$z\,(\mathrm{m})$', 'FontSize', fs, 'Interpreter', 'latex');
view(3); % Set a 3D view

%% Conversions
ro = 45 * D2R;                   % angle by which rotate the base of quadcopter
Ri = [cos(ro) -sin(ro) 0;
      sin(ro) cos(ro)  0;
       0       0       1];     % rotation matrix to rotate the coordinates of base 
base_co = [-a/2  a/2 a/2 -a/2; % Coordinates of Base 
           -a/2 -a/2 a/2 a/2;
             0    0   0   0];
base = Ri * base_co;             % rotate base Coordinates by 45 degree 

to = linspace(0, 2*pi);
xp = r_p * cos(to);
yp = r_p * sin(to);
zp = zeros(1,length(to));

%% Design Different parts
% design the base square
drone(1) = patch([base(1,:)],[base(2,:)],[base(3,:)],'r');
drone(2) = patch([base(1,:)],[base(2,:)],[base(3,:)+H],'r');
alpha(drone(1:2),0.7);

% design 2 perpendicular legs of quadcopter 
[xcylinder, ycylinder, zcylinder] = cylinder([H/2 H/2]);
drone(3) = surface(b*zcylinder-b/2,ycylinder,xcylinder+H/2,'facecolor','b');
drone(4) = surface(ycylinder,b*zcylinder-b/2,xcylinder+H/2,'facecolor','b'); 
alpha(drone(3:4),0.6);

% design 4 cylindrical motors 
drone(5) = surface(xcylinder+b/2,ycylinder,H_m*zcylinder+H/2,'facecolor','r');
drone(6) = surface(xcylinder-b/2,ycylinder,H_m*zcylinder+H/2,'facecolor','r');
drone(7) = surface(xcylinder,ycylinder+b/2,H_m*zcylinder+H/2,'facecolor','r');
drone(8) = surface(xcylinder,ycylinder-b/2,H_m*zcylinder+H/2,'facecolor','r');
alpha(drone(5:8),0.7);

% design 4 propellers
drone(9)  = patch(xp+b/2,yp,zp+(H_m+H/2),'c','LineWidth',0.5);
drone(10) = patch(xp-b/2,yp,zp+(H_m+H/2),'c','LineWidth',0.5);
drone(11) = patch(xp,yp+b/2,zp+(H_m+H/2),'p','LineWidth',0.5);
drone(12) = patch(xp,yp-b/2,zp+(H_m+H/2),'p','LineWidth',0.5);
alpha(drone(9:12),0.3);

%% Create a group object and parent surface
combinedobject = hgtransform('parent', gca);
set(drone,'parent', combinedobject);

for i = 1:length(x)
    ba = plot3(x(1:i), y(1:i), z(1:i), 'b:', 'LineWidth', 1.5);
   
    translation = makehgtform('translate', [x(i) y(i) z(i)]);
    rotation1 = makehgtform('xrotate', (pi/180)*(roll(i)));
    rotation2 = makehgtform('yrotate', (pi/180)*(pitch(i)));
    rotation3 = makehgtform('zrotate', (pi/180)*yaw(i));
    
    set(combinedobject,'matrix', translation * rotation3 * rotation2 * rotation1);
    
    % Обновляем график каждые 10 итераций для повышения скорости
    if mod(i, 10000000) == 0 
        drawnow;
    end
    pause(0.00001);
end

end
