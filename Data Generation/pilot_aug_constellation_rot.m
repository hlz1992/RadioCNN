function [new_symbols, new_tags] = pilot_aug_constellation_rot(old_symbol, old_tag, I_mat)
rotation_cast_table = [2, 4, 1, 3];
new_symbols = zeros(4, length(old_symbol));
new_tags = zeros(4, size(I_mat, 2));

new_symbols(1, :) = old_symbol;
new_tags(1, :) = I_mat(:, old_tag).';
this_tag = old_tag;
for n = 2:4
    new_symbols(n, :) = new_symbols(n-1, :) * exp(1j * pi / 2);
    this_tag = rotation_cast_table(this_tag);
    new_tags(n, :) = I_mat(:, this_tag).';
end

end