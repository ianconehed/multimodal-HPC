clear all
close all

rng(1)


N = 30;
M_dim = 80;
M = M_dim^2;
t_max = 1440;
rho = .0005;
c = 3;
b = 5;
alpha = 1;
dt = 1;

[LTP,LTD,LTP_J,LTD_J] = deal(zeros(M,t_max));

tau_p = 200; %time constant for LTP trace
tau_d = 500; %time constant for LTD trace
eta_p = 2; %activation constant for LTP trace
eta_d = 20; %activation constant for LTD trace
eta_J = .0006; %learning rate
max_p = 1.1; %maximum for LTP trace
max_d = 1; %maximum for LTD trace
tau_P = 10; %time constant for plateau potential
p_mag = 1; %magnitude of plateau potential
p_time = 180;
P = zeros(1,t_max);

R1 = rotx(2);
R2 = rotx(4);
R3 = rotx(180);

data = randn(3,N);

step_var = 0*.005;
step_vec_init = zeros(3,N);
step_vec_init(1,:) = t_max*step_var*ones(1,N)/2;

data_vec = zeros(3,N,t_max);
data_vec(:,:,1) = data - step_vec_init;

W = sprandn(M,N*3,rho);
W = abs(W) + 0*randn(M,N*3);
J = 0*abs(randn(1,M));

sp_t = zeros(M,t_max);
V_t = zeros(1,t_max);

step_vec = zeros(3,N);
step_vec(1,:) = step_var*ones(1,N);

dJ = 0;
figure('Position',[300 300 900 900])
for t = 2:t_max
    data_resize = reshape(data_vec(:,:,t-1),[N*3,1]);
    sp_t(:,t-1) = (tanh(b*W*data_resize-c) + 1)/2;
%     sp_t(sp_t(:,t-1)<.02,t-1) = 0;
    if t == 2
        V_t(:) = alpha*(J*sp_t(:,t-1));
    else 
         if t == p_time
            P(t) = P(t-1) + p_mag;
        else
            P(t) = P(t-1) - P(t-1)*(dt/tau_P);
        end
        if (P(t)>.1)
            dJ = dJ + eta_J*P(t)*sp_t(:,t-1); %weight update
            J = J + dJ';
            J(J<0) = 0;
        end
        
        V_t(t-1) = alpha*(J*sp_t(:,t-1)); %ramp amplitude at lap "lap"
    end
    
    
    if t > t_max/2
        data_vec(:,:,t) = R2*(data_vec(:,:,t-1) + step_vec);
    elseif t == t_max/2
        data_vec(:,:,t) = R1*(data_vec(:,:,t-1) + step_vec);
        data_vec(:,(2*N/3+1):N,t) = R3*(data_vec(:,(2*N/3+1):N,t-1));
    else
        data_vec(:,:,t) = R1*(data_vec(:,:,t-1) + step_vec);
    end
    
    
    
    subplot(2,2,1)
    scatter3(data_vec(1,:,t-1),data_vec(2,:,t-1),data_vec(3,:,t-1))
    xlim([-4 - t_max*step_var/2 4 + t_max*step_var/2])
    zlim([-4 4])
    ylim([-4 4])
    title('input')
    
    subplot(2,2,2)
    sp_t_resize = reshape(sp_t(:,t-1),[M_dim,M_dim]);
    imagesc(sp_t_resize)
    ax = gca;
    ax.CLim = [0 1];
    title('Sparse projections from input')
    colorbar
    
    subplot(2,1,2)
    plot(V_t(1:t-1))
    ylabel('Firing rate (AU)')
    xlabel('time (AU)')
    xlim([0 t_max])
    title('V(t)')

    drawnow
end