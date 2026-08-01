function imputation_result = ekf_imputation( ...
    residual_result, detection_result, ...
    x_pred, P_pred, H, R)
%EKF_IMPUTATION EKF update and pseudo-measurement imputation.

required_residual_fields = { ...
    'measurement_data', 'predicted_measurement', ...
    'missing_mask', 'available_mask'};

for i = 1:numel(required_residual_fields)
    if ~isfield(residual_result, required_residual_fields{i})
        error('residual_result.%s is required.', ...
            required_residual_fields{i});
    end
end

if ~isfield(detection_result, 'measurement_error_mask')
    error('detection_result.measurement_error_mask is required.');
end

measurement_data = residual_result.measurement_data;
predicted_measurement = residual_result.predicted_measurement;
missing_mask = logical(residual_result.missing_mask);
measurement_error_mask = ...
    logical(detection_result.measurement_error_mask);

if isvector(x_pred)
    x_pred = x_pred(:);
end

[m, n_sample] = size(measurement_data);
[n_state, n_state_sample] = size(x_pred);

if n_state_sample ~= n_sample
    error('x_pred must have the same number of samples as measurement_data.');
end
if ~isequal(size(measurement_error_mask), [m, n_sample])
    error('measurement_error_mask must match measurement_data.');
end

imputation_mask = missing_mask | measurement_error_mask;
update_mask = logical(residual_result.available_mask) ...
    & ~measurement_error_mask;

if any(imputation_mask & ~isfinite(predicted_measurement), 'all')
    error('predicted_measurement must be finite at imputed positions.');
end

imputed_measurement = measurement_data;
imputed_measurement(imputation_mask) = ...
    predicted_measurement(imputation_mask);

x_updated = zeros(n_state, n_sample);
P_updated = zeros(n_state, n_state, n_sample);
innovation = nan(m, n_sample);
kalman_gain = zeros(n_state, m, n_sample);

for k = 1:n_sample
    Hk = select_matrix(H, k, m, n_state, n_sample, 'H');
    Pk = select_matrix(P_pred, k, n_state, n_state, n_sample, 'P_pred');
    Rk = select_matrix(R, k, m, m, n_sample, 'R');

    used = update_mask(:, k);

    if any(used)
        H_used = Hk(used, :);
        R_used = Rk(used, used);
        innovation_used = measurement_data(used, k) ...
            - predicted_measurement(used, k);

        S = H_used * Pk * H_used' + R_used;
        S = (S + S') / 2;

        if rcond(S) < 1e-12
            K = Pk * H_used' * pinv(S);
        else
            K = (Pk * H_used') / S;
        end

        x_updated(:, k) = x_pred(:, k) + K * innovation_used;

        I = eye(n_state);
        A = I - K * H_used;
        Pk_updated = A * Pk * A' + K * R_used * K';
        P_updated(:, :, k) = (Pk_updated + Pk_updated') / 2;

        innovation(used, k) = innovation_used;
        kalman_gain(:, used, k) = K;
    else
        x_updated(:, k) = x_pred(:, k);
        P_updated(:, :, k) = Pk;
    end
end

imputation_result.measurement_data = measurement_data;
imputation_result.predicted_measurement = predicted_measurement;
imputation_result.imputed_measurement = imputed_measurement;
imputation_result.missing_mask = missing_mask;
imputation_result.measurement_error_mask = measurement_error_mask;
imputation_result.imputation_mask = imputation_mask;
imputation_result.update_mask = update_mask;
imputation_result.x_pred = x_pred;
imputation_result.P_pred = P_pred;
imputation_result.x_updated = x_updated;
imputation_result.P_updated = P_updated;
imputation_result.innovation = innovation;
imputation_result.kalman_gain = kalman_gain;
imputation_result.summary = table( ...
    (1:n_sample)', ...
    sum(missing_mask, 1)', ...
    sum(measurement_error_mask, 1)', ...
    sum(imputation_mask, 1)', ...
    sum(update_mask, 1)', ...
    'VariableNames', { ...
        'Sample', 'MissingCount', 'ErrorCount', ...
        'ImputedCount', 'UpdateCount'});
end

function matrix_k = select_matrix( ...
    data, k, n_row, n_col, n_sample, name)

if ismatrix(data)
    if ~isequal(size(data), [n_row, n_col])
        error('%s must be %d-by-%d or %d-by-%d-by-N.', ...
            name, n_row, n_col, n_row, n_col);
    end
    matrix_k = data;
elseif ndims(data) == 3
    if size(data, 1) ~= n_row ...
            || size(data, 2) ~= n_col ...
            || size(data, 3) ~= n_sample
        error('%s has incompatible dimensions.', name);
    end
    matrix_k = data(:, :, k);
else
    error('%s has incompatible dimensions.', name);
end
end
