
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% ABHISHEK MEENA
% Depatment of Electrical Engineering 
% IIT KANPUR

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;

%%
%  Part-1   TRAINING
%% intializing the parameters
neu = 10*10;
theta_gamma = rand(neu,2);
h_gamma = zeros(neu,1);
A_gamma = cell(neu,1);
w_gamma = zeros(neu,2);
sigma_ini = .1;
sigma_final = .1;
eta_ini = .1;
eta_fin = .1;

%% finding A_gamma and W_gamma
for i = 1:neu
    A_gamma{i} = [-sin(theta_gamma(i,1)), -sin(theta_gamma(i,2)); cos(theta_gamma(i,1)), cos(theta_gamma(i,2))];
    A_gamma{i} = (A_gamma{i})^(-1);
    w_gamma(i,:) = (A_gamma{i}*theta_gamma(i,:)')';
  
end
% w_gamma = rand(neu,2);
error = zeros(100);
% training_data =rand(200,2);

%% generating training data
x0=0; % x0 an y0 center coordinates
y0=0;  
radius=1.5;  % radius
angle=-pi:0.1:pi;
angl=angle(randperm(numel(angle),15));
r=rand(1,15)*radius;
x=r.*cos(angl)+x0;
y=r.*sin(angl)+y0;
xc=radius.*cos(angle)+x0;
yc=radius.*sin(angle)+y0;
training_data(:,1)=xc';
training_data(:,2)=yc';
%%
figure(1)
scatter(w_gamma(:,1),w_gamma(:,2),'g');
hold on
scatter(training_data(:,1),training_data(:,2),'r');
hold off
title('training data and neurons plot');
%% mapping function
[I,J] = ind2sub([10, 10], 1:100);

for i = 1:100
    for k = 1:63
        ut = training_data(k,:)';
        %% computing winning neuron
        for k=1:100
            dist_mat_target(k,1) = norm(ut-w_gamma(k,:)');
        end
        [value win_neu_index] = min(dist_mat_target);
        sigma = sigma_ini*(sigma_final/sigma_ini)^(i/100);
        eta = eta_ini*(eta_fin/eta_ini)^(i/100);
        win_neu = [ I(win_neu_index) J(win_neu_index) ];
        %% h_gamma
        for k=1:100
            gamma = [I(k) J(k)];
            h_gamma(k,1)= exp(-((norm(win_neu-gamma))/(2*sigma*sigma)));
        end
        s = sum(h_gamma);
        %% theta0_out and theta1_OUT
        theta0_out = zeros(2,1);
        for j = 1:neu
            theta0_out = theta0_out + h_gamma(j)*(theta_gamma(j,:)' + A_gamma{j}*(ut - w_gamma(j,:)'));
        end
        theta0_out = theta0_out/s;
        v0 = forw_kinm(theta0_out);
        theta1_out = zeros(2,1);
        for j = 1:neu
            theta1_out = theta1_out + h_gamma(j)*A_gamma{j}*(ut-v0);
        end
        theta1_out = theta0_out + theta1_out/s;
        v1 = forw_kinm(theta1_out);
        delt_v = v1 - v0;
        del_theta = theta1_out - theta0_out;
        del_theta_gamma = zeros(2,1);
        del_A_gamma = zeros(2,1);
        %% UPDATION
        for j = 1:neu
            del_theta_gamma = del_theta_gamma + h_gamma(j)*(theta_gamma(j,:)' + A_gamma{j}*(v0 - w_gamma(j,:)'));
            del_A_gamma = del_A_gamma + h_gamma(j)* A_gamma{j}* delt_v;
        end
        del_theta_gamma = del_theta_gamma/s;
        del_A_gamma = del_A_gamma/s;
        for j = 1:neu
            w_gamma(j,:) = w_gamma(j,:) + eta*h_gamma(j)*(ut' - w_gamma(j,:));
            theta_gamma(j,:) = theta_gamma(j,:) + eta*( h_gamma(j)/s*(theta0_out - del_theta_gamma))';
            A_gamma{j} = A_gamma{j} + eta*h_gamma(j)/s/(norm(delt_v)^2)*(del_theta - del_A_gamma)*delt_v';
        end
    end
    error(i) = norm(v1 - ut);
end
figure(2)
plot(error);
title('ERROR PLOT');
xlabel('iterations');
ylabel('Error');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

% Part-2 Verifying using the given input


u=[0.0,1.414; 1.414,0.0; 1.0,1.0; -1.0,-1.0; 0.8,1.2]

for t =1:5
    ut=u(t,:)';
    for k=1:100
        dist_mat_target(k,1) = norm(ut-w_gamma(k,:)');
    end
    [value win_neu_index] = min(dist_mat_target);
    
    win_neu = [ I(win_neu_index) J(win_neu_index) ];
    %% h_gamma
    for k=1:100
        gamma = [I(k) J(k)];
        h_gamma(k,1)= exp(-((norm(win_neu-gamma))/(2*sigma*sigma)));
    end
    s = sum(h_gamma);
    %% theta0_out and theta1_OUT
    theta0_out = zeros(2,1);
    for j = 1:neu
        theta0_out = theta0_out + h_gamma(j)*(theta_gamma(j,:)' + A_gamma{j}*(ut - w_gamma(j,:)'));
    end
    theta0_out = theta0_out/s;
    v0 = forw_kinm(theta0_out);
    theta1_out = zeros(2,1);
    for j = 1:neu
        theta1_out = theta1_out + h_gamma(j)*A_gamma{j}*(ut-v0);
    end
    theta1_out = theta0_out + theta1_out/s;
    v1 = forw_kinm(theta1_out);
    part1_out_XY(:,t)=v1;
    part1_out_theta(:,t)=theta1_out;
end
input = u
x_y_coordinates = part1_out_XY'
theta= part1_out_theta'





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

% Part-3 Tracking a given  circle of radius 1.5 metre
%%

u=[1.5,1.5/1.414,0.0,-1.5/1.414,-1.5,-1.5/1.414,0,1.5/1.414;0,1.5/1.414,1.5,1.5/1.414,0,-1.5/1.414,-1.5,-1.5/1.414]';

v3=zeros(2,8);
theta3=zeros(2,8);
figure(5)
ang=0:0.01:2*3.14;
xp=1.5*cos(ang);
yp=1.5*sin(ang);
plot(xp,yp);
hold on
for p=1:8
    ut=u(p,:)';
    scatter(ut(1,1),ut(2,1),'o','r');
    hold on
    min_dis=1000;
    win_ind=0;
    %calculate winning neuron
    for k=1:100
        dist_mat_target(k,1) = norm(ut-w_gamma(k,:)');
    end
    [value win_neu_index] = min(dist_mat_target);
    
    win_neu = [ I(win_neu_index) J(win_neu_index) ];
    %calculate hgamma
    for k=1:100
        gamma = [I(k) J(k)];
        h_gamma(k,1)= exp(-((norm(win_neu-gamma))/(2*sigma*sigma)));
    end
    s = sum(h_gamma);
    
    theta0_out = zeros(2,1);
    for j = 1:neu
        theta0_out = theta0_out + h_gamma(j)*(theta_gamma(j,:)' + A_gamma{j}*(ut - w_gamma(j,:)'));
    end
    theta0_out = theta0_out/s;
    v0 = forw_kinm(theta0_out);
    theta1_out = zeros(2,1);
    for j = 1:neu
        theta1_out = theta1_out + h_gamma(j)*A_gamma{j}*(ut-v0);
    end
    theta1_out = theta0_out + theta1_out/s;
    v1 = forw_kinm(theta1_out);
    
    
    v3(:,p)=v1;
    %plot
    plot([0 cos(theta1_out(1,1))],[0 sin(theta1_out(1,1))]);
    hold on
    plot([cos(theta1_out(1,1)) cos(theta1_out(1,1))+cos(theta1_out(2,1))],[sin(theta1_out(1,1)) sin(theta1_out(1,1))+sin(theta1_out(2,1))]);
    hold on
    title('Tracking a given  circle of radius 1.5 metre');
end


















