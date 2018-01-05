close all, clear all, clc;

%%
load('..\CNN_layer_outputs.mat');
load('..\conv_chan_data_AUG_idSNR_5.mat');

figure;

for id_sample = 1:size(conv1_flat_ou, 1)
    
    subplot(4, 1, 1);
    plot(conv1_flat_ou(id_sample, :));
    set(gca, 'YLim', [0, 5]);
    
    subplot(4, 1, 2);
    plot(fc1_ou(id_sample, :));
    set(gca, 'YLim', [0, 2]);
    
    subplot(4, 1, 3);
    stem(fc2_ou(id_sample, :));
    set(gca, 'YLim', [0, 0.5]);
    
    subplot(4, 1, 4);
    hold off; stem(cnn_ou(id_sample, :));
    hold on; stem(test_tag(id_sample, :), 'r');
    set(gca, 'YLim', [0, 1.0]);
    
    xlabel(['Sample ID = ', num2str(id_sample)]);
    
    pause(0.5)
end