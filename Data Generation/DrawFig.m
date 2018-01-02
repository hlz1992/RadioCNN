close all, clear all, clc;

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