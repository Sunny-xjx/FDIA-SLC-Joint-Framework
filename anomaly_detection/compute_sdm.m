function sdm_result = compute_sdm( ...
    x_wls, P_wls, imputation_result, gamma)
%COMPUTE_SDM Covariance-normalized WLS-EKF state deviation measure.

if nargin < 4 || isempty(gamma)
    gamma = 5;
end

if ~isfield(imputation_result, 'x_pred') ...
        || ~isfield(imputation_result, 'P_pred')
    error('imputation_result.x_pred and P_pred are required.');
end

x_pred = imputation_result.x_pred;
P_pred = imputation_result.P_pred;

if isvector(x_wls)
    x_wls = x_wls(:);
end
if isvector(x_pred)
    x_pred = x_pred(:);
end

if ~isequal(size(x_wls), size(x_pred))
    error('x_wls and x_pred must have the same size.');
end

validateattributes(gamma, {'numeric'}, ...
    {'real', 'finite', 'scalar', 'nonnegative'});

[n_state, n_sample] = size(x_wls);
epsilon = 1e-12;

P_wls_diag = covariance_diagonal( ...
    P_wls, n_state, n_sample, 'P_wls');
P_pred_diag = covariance_diagonal( ...
    P_pred, n_state, n_sample, 'P_pred');

variance_sum = max(P_wls_diag + P_pred_diag, 0);

sdm = abs(x_wls - x_pred) ...
    ./ sqrt(variance_sum + epsilon);

[max_sdm, trigger_state] = max(sdm, [], 1);
detected = max_sdm > gamma;

status = strings(n_sample, 1);
status(~detected) = "Normal";
status(detected) = "FDIA/SLC candidate";

sdm_result.x_wls = x_wls;
sdm_result.x_pred = x_pred;
sdm_result.P_wls_diag = P_wls_diag;
sdm_result.P_pred_diag = P_pred_diag;
sdm_result.sdm = sdm;
sdm_result.max_sdm = max_sdm;
sdm_result.trigger_state = trigger_state;
sdm_result.detected = detected;
sdm_result.gamma = gamma;
sdm_result.summary = table( ...
    (1:n_sample)', ...
    max_sdm', ...
    trigger_state', ...
    detected', ...
    status, ...
    'VariableNames', { ...
        'Sample', 'MaxSDM', 'TriggerState', ...
        'Detected', 'Status'});
end

function diagonal_data = covariance_diagonal( ...
    covariance_data, n_state, n_sample, name)

if ismatrix(covariance_data)
    if ~isequal(size(covariance_data), [n_state, n_state])
        error('%s must be n-by-n or n-by-n-by-N.', name);
    end
    diagonal_data = repmat( ...
        diag(covariance_data), 1, n_sample);
elseif ndims(covariance_data) == 3
    if size(covariance_data, 1) ~= n_state ...
            || size(covariance_data, 2) ~= n_state ...
            || size(covariance_data, 3) ~= n_sample
        error('%s has incompatible dimensions.', name);
    end

    diagonal_data = zeros(n_state, n_sample);
    for k = 1:n_sample
        diagonal_data(:, k) = ...
            diag(covariance_data(:, :, k));
    end
else
    error('%s has incompatible dimensions.', name);
end
end
