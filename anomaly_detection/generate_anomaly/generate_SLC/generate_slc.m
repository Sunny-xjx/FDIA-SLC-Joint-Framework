clc;
clear;
close all;

load('load_values_all.mat');

power_factor = 0.95;
phi = acos(power_factor);

n_sample = 6000;
N_SLC = 1;
eta_min = -0.30;
eta_max = -0.10;
noise_level = 1 / 300;

mpc_base = loadcase('case30');
mpopt = mpoption('verbose', 0, 'out.all', 0);

slack_bus = 1;
pv_buses = [2, 13, 22, 23, 27];

all_buses = mpc_base.bus(:, 1);
pq_buses = setdiff(all_buses, [slack_bus, pv_buses]);

n_load_profile = min([20, size(combined_data, 1), numel(pq_buses)]);
load_buses = pq_buses(1:n_load_profile);
zero_buses = setdiff(pq_buses, load_buses);

n_sample = min(n_sample, size(combined_data, 2));

if N_SLC < 1 || N_SLC > numel(load_buses)
    error('N_SLC must be between 1 and the number of load buses.');
end

n_bus = size(mpc_base.bus, 1);
n_measurement = 3 * n_bus - 2;
load_scale = 1 / (mpc_base.baseMVA * 100);

SLC_measurement_nominal = nan(n_measurement, n_sample);
SLC_measurement = nan(n_measurement, n_sample);
measurement_noise = nan(n_measurement, n_sample);
sigma = nan(n_measurement, n_sample);

VM = nan(n_bus, n_sample);
VA = nan(n_bus, n_sample);
P_injection = nan(n_bus, n_sample);
Q_injection = nan(n_bus, n_sample);

Pd_nominal = nan(n_bus, n_sample);
Qd_nominal = nan(n_bus, n_sample);
Pd_slc = nan(n_bus, n_sample);
Qd_slc = nan(n_bus, n_sample);

SLC_bus_mask = false(n_bus, n_sample);
SLC_ratio = zeros(n_bus, n_sample);
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

    Pd_nominal(:, k) = mpc_k.bus(:, 3);
    Qd_nominal(:, k) = mpc_k.bus(:, 4);

    selected_index = randperm(numel(load_buses), N_SLC);
    selected_buses = load_buses(selected_index);
    eta = eta_min + (eta_max - eta_min) * rand(N_SLC, 1);

    for i = 1:N_SLC
        bus_id = selected_buses(i);
        row = mpc_k.bus(:, 1) == bus_id;

        mpc_k.bus(row, 3) = mpc_k.bus(row, 3) * (1 + eta(i));
        mpc_k.bus(row, 4) = mpc_k.bus(row, 4) * (1 + eta(i));

        SLC_bus_mask(bus_id, k) = true;
        SLC_ratio(bus_id, k) = eta(i);
    end

    Pd_slc(:, k) = mpc_k.bus(:, 3);
    Qd_slc(:, k) = mpc_k.bus(:, 4);

    results = runpf(mpc_k, mpopt);

    if ~results.success
        continue;
    end

    vm = results.bus(:, 8);
    va = results.bus(:, 9);

    Pg = accumarray( ...
        results.gen(:, 1), results.gen(:, 2), ...
        [n_bus, 1], @sum, 0);

    Qg = accumarray( ...
        results.gen(:, 1), results.gen(:, 3), ...
        [n_bus, 1], @sum, 0);

    p_inj = Pg - results.bus(:, 3);
    q_inj = Qg - results.bus(:, 4);

    z_hat_slc = [ ...
        vm;
        p_inj(2:end);
        q_inj(2:end)];

    sigma_k = abs(noise_level .* z_hat_slc);
    sigma_k(sigma_k < 1e-8) = 1e-8;

    epsilon_k = sigma_k .* randn(n_measurement, 1);
    z_slc = z_hat_slc + epsilon_k;

    SLC_measurement_nominal(:, k) = z_hat_slc;
    SLC_measurement(:, k) = z_slc;
    measurement_noise(:, k) = epsilon_k;
    sigma(:, k) = sigma_k;

    VM(:, k) = vm;
    VA(:, k) = va;
    P_injection(:, k) = p_inj;
    Q_injection(:, k) = q_inj;

    success_mask(k) = true;

    if mod(k, 500) == 0
        fprintf('Generated %d / %d samples.\n', k, n_sample);
    end
end

valid = success_mask;

SLC_measurement_nominal = SLC_measurement_nominal(:, valid);
SLC_measurement = SLC_measurement(:, valid);
measurement_noise = measurement_noise(:, valid);
sigma = sigma(:, valid);

VM = VM(:, valid);
VA = VA(:, valid);
P_injection = P_injection(:, valid);
Q_injection = Q_injection(:, valid);

Pd_nominal = Pd_nominal(:, valid);
Qd_nominal = Qd_nominal(:, valid);
Pd_slc = Pd_slc(:, valid);
Qd_slc = Qd_slc(:, valid);

SLC_bus_mask = SLC_bus_mask(:, valid);
SLC_ratio = SLC_ratio(:, valid);
sample_index = find(valid);

measurement_data = SLC_measurement;

output_file = sprintf('slc_IEEE30_N%d_data.mat', N_SLC);

save(output_file, 'measurement_data');

fprintf('Saved %d valid SLC samples to %s.\n', ...
    size(measurement_data, 2), output_file);
