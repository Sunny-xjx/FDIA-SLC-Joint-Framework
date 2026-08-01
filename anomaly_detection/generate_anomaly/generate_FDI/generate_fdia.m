clc;
clear;
close all;

load('load_values_all.mat');

power_factor = 0.95;
phi = acos(power_factor);

n_sample = 6000;
N_theta = 1;
N_V = 1;
s_theta = 0.10;
s_V = 0.05;
noise_level = 1 / 300;

mpc_base = loadcase('case30');
mpopt = mpoption('verbose', 0, 'out.all', 0);

slack_bus = 1;
pv_buses = [2, 13, 22, 23, 27];

all_buses = mpc_base.bus(:, 1);
non_slack_buses = setdiff(all_buses, slack_bus);
pq_buses = setdiff(all_buses, [slack_bus, pv_buses]);

n_load_profile = min([20, size(combined_data, 1), numel(pq_buses)]);
load_buses = pq_buses(1:n_load_profile);
zero_buses = setdiff(pq_buses, load_buses);

n_sample = min(n_sample, size(combined_data, 2));

if N_theta < 0 || N_theta > numel(non_slack_buses)
    error('N_theta is outside the valid range.');
end

if N_V < 0 || N_V > numel(non_slack_buses)
    error('N_V is outside the valid range.');
end

if N_theta == 0 && N_V == 0
    error('At least one voltage-state component must be attacked.');
end

n_bus = size(mpc_base.bus, 1);
n_measurement = 3 * n_bus - 2;
load_scale = 1 / (mpc_base.baseMVA * 100);

measurement_data = nan(n_measurement, n_sample);
sigma = nan(n_measurement, n_sample);
success_mask = false(1, n_sample);

for k = 1:n_sample
    mpc_k = mpc_base;

    active_load = combined_data(1:n_load_profile, k) * load_scale;
    reactive_load = tan(phi) * active_load;

    for i = 1:numel(load_buses)
        bus_id = load_buses(i);
        row = mpc_k.bus(:, 1) == bus_id;

        mpc_k.bus(row, 3) = active_load(i);
        mpc_k.bus(row, 4) = reactive_load(i);
    end

    for i = 1:numel(zero_buses)
        bus_id = zero_buses(i);
        row = mpc_k.bus(:, 1) == bus_id;

        mpc_k.bus(row, 3) = 0;
        mpc_k.bus(row, 4) = 0;
    end

    results = runpf(mpc_k, mpopt);

    if ~results.success
        continue;
    end

    v_hat = results.bus(:, 8);
    theta_hat = results.bus(:, 9);

    v_attack = v_hat;
    theta_attack = theta_hat;

    if N_theta > 0
        theta_index = randperm(numel(non_slack_buses), N_theta);
        theta_buses = non_slack_buses(theta_index);
        delta_theta = (2 * rand(N_theta, 1) - 1) * s_theta;

        for i = 1:N_theta
            bus_id = theta_buses(i);
            theta_attack(bus_id) = ...
                theta_hat(bus_id) * (1 + delta_theta(i));
        end
    end

    if N_V > 0
        voltage_index = randperm(numel(non_slack_buses), N_V);
        voltage_buses = non_slack_buses(voltage_index);
        delta_v = (2 * rand(N_V, 1) - 1) * s_V;

        for i = 1:N_V
            bus_id = voltage_buses(i);
            v_attack(bus_id) = ...
                v_hat(bus_id) * (1 + delta_v(i));
        end
    end

    z_hat = state_to_measurement(results, v_hat, theta_hat);
    z_hat_attack = state_to_measurement( ...
        results, v_attack, theta_attack);

    sigma_k = abs(noise_level .* z_hat);
    sigma_k(sigma_k < 1e-8) = 1e-8;

    epsilon_k = sigma_k .* randn(n_measurement, 1);
    z_normal = z_hat + epsilon_k;

    attack_vector = z_hat_attack - z_hat;
    z_fdia = z_normal + attack_vector;

    measurement_data(:, k) = z_fdia;
    sigma(:, k) = sigma_k;
    success_mask(k) = true;

    if mod(k, 500) == 0
        fprintf('Generated %d / %d samples.\n', k, n_sample);
    end
end

measurement_data = measurement_data(:, success_mask);
sigma = sigma(:, success_mask);

output_file = sprintf( ...
    'fdia_IEEE30_Ntheta%d_NV%d_data.mat', ...
    N_theta, N_V);

save(output_file, 'measurement_data', 'sigma');

fprintf('Saved %d valid FDIA samples to %s.\n', ...
    size(measurement_data, 2), output_file);

function measurement = state_to_measurement(mpc, vm, va)
[Ybus, ~, ~] = makeYbus( ...
    mpc.baseMVA, mpc.bus, mpc.branch);

V = vm .* exp(1j * va * pi / 180);
Sbus = V .* conj(Ybus * V) * mpc.baseMVA;

P_injection = real(Sbus);
Q_injection = imag(Sbus);

measurement = [ ...
    vm;
    P_injection(2:end);
    Q_injection(2:end)];
end
