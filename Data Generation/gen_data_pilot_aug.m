close all, clear all, clc;

rand('seed', 1);
randn('seed', 1);

%% Basic parameters
pilot_augmentation = 1;
mod = 4;
chan_len = 16;

input_dim = [2, 41];    % channeled symbols
output_dim = mod;

total_num = 11000;
pilot_num_original = 500;
sample_indices_temp = randerr(1, total_num, pilot_num_original);
sample_indices = find(sample_indices_temp > 0).';
data_num = total_num - pilot_num_original;

if (input_dim(2)-1)/2 <= chan_len
    error('Sameple dimension too small!')
end

padding_num = (input_dim(2)-1)/2;

%% Generate ISI channel & modulation mapper
h = randn(1, chan_len) + 1j * randn(1, chan_len);
h = h .* exp(-[0:chan_len-1]/4);
% figure; stem(abs(h))
h = h / norm(h);
% load('h_save.mat');
mod_mapper = qammod([0:mod-1], mod);
mod_mapper = mod_mapper / norm(mod_mapper) * sqrt(mod);

%% Quantization configuration
quan_bits_num = 5;
quan_max_amp = 2;
quan_switch = 0;

%% Generate all data
I_mat = eye(mod);
data_symbols = randi(mod, 1, total_num);
data_chan_in = [zeros(1, padding_num), mod_mapper(data_symbols), zeros(1, padding_num)];
data_chan_out = conv(data_chan_in, h);

if quan_switch == 1
    temp1 = real(data_chan_out);
    temp2 = imag(data_chan_out);
    data_chan_out = func_quan(temp1, quan_max_amp, quan_bits_num) + ...
        1j * func_quan(temp2, quan_max_amp, quan_bits_num);
end

test_data = zeros(total_num, 2 * input_dim(2));
test_tag = zeros(total_num, output_dim);
for id_data = 1:total_num
    sym_index = padding_num + id_data;
    temp = data_chan_out(sym_index-padding_num:sym_index+padding_num);
    test_data(id_data, :) = [real(temp), imag(temp)];
    test_tag(id_data, :) = I_mat(:, data_symbols(id_data)).';
end

% sample_data
sample_data = test_data(sample_indices, :);
sample_tag = test_tag(sample_indices, :);

% test_data
test_data(sample_indices, :) = [];
test_tag(sample_indices, :) = [];

if size(test_data, 1) ~= data_num
    error('Wrong size!');
end

%% Add noise & save
SNRdBRng = linspace(0, 7, 5);

test_data_bak = test_data;
sample_data_bak = sample_data;
sample_tag_bak = sample_tag;

for id_SNR = 1:length(SNRdBRng)
    SNR = 10^(SNRdBRng(id_SNR) / 10);
    
    test_data = test_data_bak + randn(size(test_data_bak)) / (2 * SNR);
    sample_data = sample_data_bak + randn(size(sample_data_bak)) / (2 * SNR);
    sample_tag = sample_tag_bak;
    
    if pilot_augmentation
        %%  Apply pilot augmentation
        sample_data_aug = zeros(20 * pilot_num_original, size(sample_data, 2));
        sample_tag_aug = zeros(20 * pilot_num_original, size(sample_tag, 2));
        aug_count = 1;
        
        for id_sample = 1:size(sample_data, 1)
            this_data = sample_data(id_sample, :);
            this_data = this_data(1:length(this_data)/2) + ...
                1j * this_data(length(this_data)/2+1:end);
            this_tag = find(sample_tag(id_sample, :) > 0);
            
            % Step 1: rotation
            [new_symbols_0, new_tags_0] = pilot_aug_constellation_rot(this_data, this_tag, I_mat);
            
            sample_data_aug(aug_count:aug_count+3, :) = [real(new_symbols_0), imag(new_symbols_0)];
            sample_tag_aug(aug_count:aug_count+3, :) = new_tags_0;
            aug_count = aug_count+4;
            
            % Step 2: noisy
            rep_constant = 2;
            new_symbols_1 = repmat(new_symbols_0, rep_constant, 1);
            new_tags_1 = repmat(new_tags_0, rep_constant, 1);
            avg_power = mean(mean(abs(new_symbols_1).^2));
            new_symbols_1_tmp = new_symbols_1;
            new_symbols_1 = new_symbols_1 + (randn(size(new_symbols_1, 1), size(new_symbols_1, 2)) ...
                + 1j * randn(size(new_symbols_1, 1), size(new_symbols_1, 2))) * sqrt(avg_power/2 * 1e-2);
            
            sample_data_aug(aug_count:aug_count+4*rep_constant-1, :) = [real(new_symbols_1), imag(new_symbols_1)];
            sample_tag_aug(aug_count:aug_count+4*rep_constant-1, :) = new_tags_1;
            
            aug_count = aug_count + 4*rep_constant;
        end
        
        pilot_num = aug_count - 1;
        sample_data_aug = sample_data_aug(1:pilot_num, :);
        sample_tag_aug = sample_tag_aug(1:pilot_num, :);
        
        sample_data = sample_data_aug;
        sample_tag = sample_tag_aug;
        
        file_name = ['..\conv_chan_data_AUG_idSNR_', num2str(id_SNR), '.mat'];
        save(file_name, 'test_data', 'test_tag', 'sample_data', 'sample_tag');
    else
        pilot_num = size(sample_data, 1);
        file_name = ['..\conv_chan_data_idSNR_', num2str(id_SNR), '.mat'];
        save(file_name, 'test_data', 'test_tag', 'sample_data', 'sample_tag');
    end
end

disp(['Number of pilots after augmentation: ', num2str(pilot_num)]);

%% Write parameter files
fid = fopen('..\matlab_params.py', 'w');
fprintf(fid, 'matlab_mod = %d\n', mod);
fprintf(fid, 'matlab_chan_len = %d\n', chan_len);
fprintf(fid, 'matlab_input_dim = [%d, %d]\n', input_dim(1), input_dim(2));
fprintf(fid, 'matlab_output_dim = %d\n', output_dim);
fprintf(fid, 'matlab_sample_num = %d\n', pilot_num);
fprintf(fid, 'matlab_data_num = %d\n', data_num);
fclose(fid);


