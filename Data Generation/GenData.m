close all, clear all, clc;

rand('seed', 1);
randn('seed', 1);

%% Basic parameters
SER_eval = 0;
mod = 4;
chan_len = 16;

input_dim = [2, 41];    % channeled symbols
output_dim = mod;

total_num = 11000;
sample_indices_temp = randerr(1, total_num, 1e3);
sample_indices = find(sample_indices_temp > 0).';
pilot_num = length(sample_indices);
data_num = total_num - pilot_num;

if (input_dim(2)-1)/2 <= chan_len
    error('Sameple dimension too small!')
end

padding_num = (input_dim(2)-1)/2;

%% Write parameter files
fid = fopen('..\matlab_params.py', 'w');
fprintf(fid, 'matlab_mod = %d\n', mod);
fprintf(fid, 'matlab_chan_len = %d\n', chan_len);
fprintf(fid, 'matlab_input_dim = [%d, %d]\n', input_dim(1), input_dim(2));
fprintf(fid, 'matlab_output_dim = %d\n', output_dim);
fprintf(fid, 'matlab_sample_num = %d\n', pilot_num);
fprintf(fid, 'matlab_data_num = %d\n', data_num);
fclose(fid);

%% Generate ISI channel & modulation mapper
h = randn(1, chan_len) + 1j * randn(1, chan_len);
h = h .* exp(-[0:chan_len-1]/4);
figure; stem(abs(h))
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

% sample indices

% sample_data
sample_data = test_data(sample_indices, :);
sample_tag = test_tag(sample_indices, :);

% test_data
test_data(sample_indices, :) = [];
test_tag(sample_indices, :) = [];

if size(test_data, 1) ~= data_num
    error('Wrong size!');
end

%% Save
save('..\conv_chan_data.mat', 'test_data', 'test_tag', 'sample_data', 'sample_tag');

%% SER Evaluation
if SER_eval
    SNRdBRng = linspace(0, 7, 5);
    test_data_num = 1e3;
    trial_num = 1e2;

    SER_mmse = 0 * SNRdBRng;
    SER_ls = 0 * SNRdBRng;
    for id_SNR = 1:length(SNRdBRng)
        id_SNR
        SNR = 10^(SNRdBRng(id_SNR) / 10);

        pilot_noise_out = pilot_chan_out + ...
            (randn(size(pilot_chan_out)) + 1j*randn(size(pilot_chan_out))) / ...
            (2 * SNR);
        pilot_noise_out = pilot_noise_out(padding_num+1:end);

        % Channel estimation
        y = pilot_noise_out(1:pilot_num+chan_len-1).';
        h_esti = h.';        % TEST
        H_esti = zeros(test_data_num+chan_len-1, test_data_num);
        for id_data = 1:test_data_num
            H_esti(id_data:id_data+chan_len-1, id_data) = h_esti;
        end
        M_LS = inv(H_esti' * H_esti) * H_esti';
        M_MMSE = H_esti' * inv(H_esti * H_esti' + 1/SNR * eye(size(H_esti, 1)));

        % Equalization
        for idTrial = 1:trial_num
            test_symbols = randi(mod, 1, test_data_num);
            test_chan_in = [mod_mapper(test_symbols), zeros(1, padding_num)];
            test_chan_out = conv(test_chan_in, h);

            if quan_switch == 1
                temp1 = real(test_chan_out);
                temp2 = imag(test_chan_out);
                test_chan_out = func_quan(temp1, quan_max_amp, quan_bits_num) + ...
                    1j * func_quan(temp2, quan_max_amp, quan_bits_num);
            end

            test_noise_out = test_chan_out + ...
                (randn(size(test_chan_out)) + 1j*randn(size(test_chan_out))) / ...
                (2 * SNR);
            test_noise_out = test_noise_out(1:test_data_num+chan_len-1);

            rec_symbols = M_MMSE * test_noise_out.';
            temp = abs(repmat(rec_symbols, 1, mod) - repmat(mod_mapper, test_data_num, 1)).^2;
            [~, rec_idx] = min(temp.');
            SER_mmse(id_SNR) = SER_mmse(id_SNR) + sum(rec_idx ~= test_symbols)/test_data_num/trial_num;

            rec_symbols = M_LS * test_noise_out.';
            temp = abs(repmat(rec_symbols, 1, mod) - repmat(mod_mapper, test_data_num, 1)).^2;
            [~, rec_idx] = min(temp.');
            SER_ls(id_SNR) = SER_ls(id_SNR) + sum(rec_idx ~= test_symbols)/test_data_num/trial_num;
        end
    end

    %% Save
    save('..\SER_benchmark.mat', 'SNRdBRng', 'SER_ls', 'SER_mmse')

    %% 
    load('..\SER_benchmark.mat');
    figure; hold on;
    plot(SNRdBRng, SER_mmse, 'rx--');
    plot(SNRdBRng, SER_ls, 'r<-');
    load('..\RadioNN_performance.mat');
    plot(SNR_DB_RANGE, SER, 'bo-');
    legend('mmse', 'ls');
    set(gca, 'YScale', 'log');
    xlabel('SNR (dB)');
    ylabel('Symbol Error Rate');
    grid on;
end



