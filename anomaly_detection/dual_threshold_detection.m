function detection_result = dual_threshold_detection( ...
    residual_result, T1, T2)
%DUAL_THRESHOLD_DETECTION Residual amplitude-angle screening.

if nargin < 2 || isempty(T1)
    T1 = 0.93;
end
if nargin < 3 || isempty(T2)
    T2 = 10;
end

required_fields = {'residual', 'sigma', 'available_mask'};
for i = 1:numel(required_fields)
    if ~isfield(residual_result, required_fields{i})
        error('residual_result.%s is required.', required_fields{i});
    end
end

residual = residual_result.residual;
sigma = residual_result.sigma;
available_mask = logical(residual_result.available_mask);

if ~isequal(size(residual), size(sigma), size(available_mask))
    error('residual, sigma, and available_mask must have the same size.');
end

validateattributes(T1, {'numeric'}, ...
    {'real', 'finite', 'scalar', 'nonnegative'});
validateattributes(T2, {'numeric'}, ...
    {'real', 'finite', 'scalar', '>=', 0, '<=', 90});

[m, n_sample] = size(residual);

normalized_residual = zeros(m, n_sample);
normalized_residual(available_mask) = ...
    residual(available_mask) ./ sigma(available_mask);

magnitude = vecnorm(normalized_residual, 2, 1);
denominator = max(magnitude, eps);

axis_ratio = abs(normalized_residual) ./ denominator;
axis_ratio = min(max(axis_ratio, 0), 1);

theta_deg = acosd(axis_ratio);
[theta_min_deg, dominant_channel] = min(theta_deg, [], 1);

region = ones(1, n_sample);
above_T1 = magnitude > T1;

region(above_T1 & theta_min_deg <= T2) = 2;
region(above_T1 & theta_min_deg > T2) = 3;

measurement_error_mask = false(m, n_sample);
region_ii_index = find(region == 2);

for k = region_ii_index
    channel = dominant_channel(k);
    if available_mask(channel, k)
        measurement_error_mask(channel, k) = true;
    end
end

error_channel = nan(1, n_sample);
error_channel(region == 2) = dominant_channel(region == 2);

status = strings(n_sample, 1);
status(region == 1) = "Normal";
status(region == 2) = "Measurement error";
status(region == 3) = "Other anomaly";

detection_result.normalized_residual = normalized_residual;
detection_result.magnitude = magnitude;
detection_result.theta_deg = theta_deg;
detection_result.theta_min_deg = theta_min_deg;
detection_result.region = region;
detection_result.error_channel = error_channel;
detection_result.measurement_error_mask = measurement_error_mask;
detection_result.T1 = T1;
detection_result.T2 = T2;
detection_result.summary = table( ...
    (1:n_sample)', ...
    magnitude', ...
    theta_min_deg', ...
    region', ...
    error_channel', ...
    status, ...
    'VariableNames', { ...
        'Sample', 'Magnitude', 'ThetaMinDeg', ...
        'Region', 'ErrorChannel', 'Status'});
end
