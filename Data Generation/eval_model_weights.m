close all, clear all, clc;

%%
load('../RadioCNN_weights.mat');
fc1_w_0 = fc1_weights;
fc2_w_0 = fc2_weights;
k0 = conv1_kernal;

%%
load('../RadioCNN_weights_nc.mat');
fc1_w_1 = fc1_weights;
fc2_w_1 = fc2_weights;
k1 = conv1_kernal;

temp = zeros(1, 8);
for i = 1:8
    temp(i) = sum(sum(abs(k0(:, :, 1, i) - k1(:, :, 1, i)).^2)) / sum(sum(abs(k0(:, :, 1, i)).^2));
end
temp
sum(sum(abs(fc1_w_0 - fc1_w_1).^2)) / sum(sum(abs(fc1_w_0).^2))
sum(sum(abs(fc2_w_0 - fc2_w_1).^2)) / sum(sum(abs(fc2_w_0).^2))

figure; hold on; 
plot(reshape(fc1_w_0, 656*256, 1));
plot(reshape(fc1_w_1, 656*256, 1), 'r');

figure; hold on; 
plot(reshape(fc2_w_0, 128*256, 1));
plot(reshape(fc2_w_1, 128*256, 1), 'r');

