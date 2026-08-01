clc;clear

PF = 0.95;       
theta = acos(PF);
n = 6000;        
N = 30;          

mpc = loadcase('case30');

load('load_values_all.mat');


slack_bus = 1;
pv_buses = [2, 13, 22, 23, 27];


all_buses = mpc.bus(:, 1);
pq_buses = setdiff(all_buses, [slack_bus, pv_buses]);

VM = zeros(30, n);      
VA = zeros(30, n);    
PF = zeros(41, n);    
GF = zeros(41, n);    
PT = zeros(41, n);    
QT = zeros(41, n);   

for j = 1:n

    load_values = combined_data(:,j);
    for i = 1:20
        bus_id = pq_buses(i);
        mpc.bus(mpc.bus(:,1) == bus_id, 3) = load_values(i) / mpc.baseMVA / 100;         % Pd
        mpc.bus(mpc.bus(:,1) == bus_id, 4) = tan(theta) * load_values(i) / mpc.baseMVA / 100;   % Qd（假设 Q/P = 0.3）
    end
    
    zero_buses = setdiff(pq_buses, pq_buses(1:20));

    for i = 1:length(zero_buses)
        bus_id = zero_buses(i);
        mpc.bus(mpc.bus(:,1) == bus_id, 3) = 0;
        mpc.bus(mpc.bus(:,1) == bus_id, 4) = 0;
    end
    
    results = runpf(mpc); 
    vm = results.bus(:, 8); 
    va = results.bus(:, 9); 
    pf = results.branch(:, 14);  
    gf = results.branch(:, 15); 
    pt = results.branch(:, 16);  
    qt = results.branch(:, 17);  

    VM(:, j) = vm;   
    VA(:, j) = va;    
    PF(:, j) = pf;  
    GF(:, j) = gf;  
    PT(:, j) = pt; 
    QT(:, j) = qt;  
end


Input = zeros(41*7, n);  
for k = 1:length(results.branch)
    num = results.branch(k, 1);
    Input(7*(k-1)+1, :) = VM(num, :).^2;   
    if ismember(num, [slack_bus, pv_buses, zero_buses']) 
        Input(7*(k-1)+2, :) = zeros(1, n);
        Input(7*(k-1)+3, :) = zeros(1, n);
    else 
        index = find(pq_buses(1:20) == num);
        Input(7*(k-1)+2, :) = combined_data(index, 1:n) / mpc.baseMVA / 100;
        Input(7*(k-1)+3, :) = theta * combined_data(index, 1:n) / mpc.baseMVA / 100;
    end
    Input(7*(k-1)+4, :) = PF(k, :);
    Input(7*(k-1)+5, :) = GF(k, :);
    Input(7*(k-1)+6, :) = PT(k, :);
    Input(7*(k-1)+7, :) = QT(k, :);
end

Labels = zeros(N*2, n);
Labels(1:N, :) = VM; 
Labels(N+1:60, :) = VA;

save('dist2_30FASE_data.mat', 'Input', 'Labels');

               