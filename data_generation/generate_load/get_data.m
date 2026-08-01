clc;clear

data = readtable('Load_history.csv');

hour_columns = data.Properties.VariableNames( ...
    startsWith(data.Properties.VariableNames, 'h'));


years = 2004:2007;   
months = 1:12;       
days = 1:20;         


for y = years
    for m = months
        for d = days
           
            target_date_data = data(data.year == y & data.month == m & data.day == d, :);

       
            if isempty(target_date_data)
                continue;
            end

     
            zone1to20_data = target_date_data(ismember(target_date_data.zone_id, 1:20), :);

            if height(zone1to20_data) < 20
                continue;
            end

            zone1to20_data = sortrows(zone1to20_data, 'zone_id');

            load_values = zone1to20_data{:, hour_columns}; 

            filename = sprintf('load_values_%04d%02d%02d.mat', y, m, d);

            if(~isempty(load_values{1,1}))  
                save(filename, 'load_values');
            end
        end
    end
end

file_list = dir('load_values_*.mat');  
num_files = length(file_list);

combined_data = zeros(20, 24 * num_files); 

for i = 1:num_files
    file_name = file_list(i).name;
    
    data = load(file_name); 
    
    current_data = data.load_values;  

    current_data = str2double(strrep(current_data, ',', ''));  
    
    combined_data(:, (i-1)*24 + 1 : i*24) = current_data;
end

disp(['拼接后的数组尺寸: ', num2str(size(combined_data))]);
