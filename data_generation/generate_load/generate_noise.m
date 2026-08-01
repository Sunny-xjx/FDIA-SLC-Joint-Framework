% 产生混合高斯噪声，拉普拉斯噪声代码 %
clc;clear
load('dist2_30FASE_data.mat');
disp('=== 电力系统测量噪声生成器 ===');
 
m = 287;       % Input的行数
n = 6000;      % Input的列数
Input_Gaussian = zeros(m, n);
Input_Laplace = zeros(m, n);

% 用户输入参数
noise_type = input('请选择噪声类型（输入"高斯"或"拉普拉斯"）：', 's');

 % 根据输入调用不同的噪声生成函数
    switch lower(noise_type)
        case '高斯'
            for i = 1:m
                for j = 1:n
                    signal = Input(i,j);
                    noisy_signal = add_bimodal_gaussian_noise(signal);
                    Input_Gaussian(i,j) = noisy_signal;
                end
            end
            
        case '拉普拉斯'
            for i = 1:m
                for j = 1:n
                    signal = Input(i,j);
                    noisy_signal = add_laplace_noise(signal);
                    Input_Laplace(i,j) = noisy_signal;
                end
            end
            
        otherwise
            error('输入错误！请选择"高斯"或"拉普拉斯"');
    end

    % save('bimodal_gaussian_data.mat', 'Input_Gaussian', 'Labels');
    save('Laplace_data_0.5.mat', 'Input_Laplace', 'Labels');