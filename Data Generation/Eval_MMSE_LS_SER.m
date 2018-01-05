close all, clear all, clc;

rand('seed', 1);
randn('seed', 1);

%% Basic parameters
mod = 4;
chan_len = 16;

input_dim = [2, 41];    % channeled symbols
output_dim = mod;

pilot_num = 500;

%% Generate ISI channel & modulation mapper
h = randn(1, chan_len) + 1j * randn(1, chan_len);
h = h .* exp(-[0:chan_len-1]/4);
h = h / norm(h);
% load('h_save.mat');
mod_mapper = qammod([0:mod-1], mod);
mod_mapper = mod_mapper / norm(mod_mapper) * sqrt(mod);

padding_num = (input_dim(2)-1)/2;
%% Generate pilot
I_mat = eye(mod);
pilot_symbols = mod_mapper(randi(mod, 1, pilot_num));
pilot_chan_in = [zeros(1, padding_num), pilot_symbols, zeros(1, padding_num)];
pilot_chan_out = conv(pilot_chan_in, h);

A_pilot = zeros(pilot_num + chan_len - 1, chan_len);
for t = 1:chan_len
    A_pilot(t:t+pilot_num-1, t) = pilot_symbols.';
end

%% SER Evaluation
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

%     h_esti = h.';        % TEST
    h_esti = A_pilot' * inv(A_pilot * A_pilot' + 1/SNR * eye(size(A_pilot, 1))) * y;

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



