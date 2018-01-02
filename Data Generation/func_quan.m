function x_ = func_quan(x, max_amp, quan_bit_num)
delta_x = max_amp / (2^(quan_bit_num - 1) - 1);
x_ = sign(x) .* floor(abs(x)/delta_x) * delta_x + delta_x/2;
end