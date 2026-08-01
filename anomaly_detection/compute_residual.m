function residual_result = compute_residual( ...
    measurement_data, predicted_measurement, sigma)
%COMPUTE_RESIDUAL Compute the EKF prediction residual.

if isvector(measurement_data)
    measurement_data = measurement_data(:);
end
if isvector(predicted_measurement)
    predicted_measurement = predicted_measurement(:);
end

if ~isequal(size(measurement_data), size(predicted_measurement))
    error('measurement_data and predicted_measurement must have the same size.');
end

[m, n_sample] = size(measurement_data);
sigma = expand_sigma(sigma, m, n_sample);

missing_mask = ~isfinite(measurement_data);
available_mask = isfinite(measurement_data) ...
    & isfinite(predicted_measurement) ...
    & isfinite(sigma) ...
    & abs(sigma) > 1e-12;

residual_raw = measurement_data - predicted_measurement;
residual = zeros(m, n_sample);
residual(available_mask) = residual_raw(available_mask);

sigma = abs(sigma);
sigma(~available_mask) = 1;
sigma(sigma < 1e-12) = 1e-12;

residual_result.measurement_data = measurement_data;
residual_result.predicted_measurement = predicted_measurement;
residual_result.sigma = sigma;
residual_result.residual_raw = residual_raw;
residual_result.residual = residual;
residual_result.missing_mask = missing_mask;
residual_result.available_mask = available_mask;
residual_result.summary = table( ...
    (1:n_sample)', ...
    sum(missing_mask, 1)', ...
    sum(available_mask, 1)', ...
    vecnorm(residual, 2, 1)', ...
    'VariableNames', { ...
        'Sample', 'MissingCount', 'AvailableCount', 'ResidualNorm'});
end

function sigma_full = expand_sigma(sigma, m, n_sample)
if isscalar(sigma)
    sigma_full = repmat(sigma, m, n_sample);
elseif isvector(sigma)
    sigma = sigma(:);
    if numel(sigma) ~= m
        error('sigma must contain one value per measurement channel.');
    end
    sigma_full = repmat(sigma, 1, n_sample);
elseif isequal(size(sigma), [m, n_sample])
    sigma_full = sigma;
else
    error('sigma must be scalar, m-by-1, or m-by-N.');
end
end
