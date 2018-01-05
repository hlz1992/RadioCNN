function [new_symbol, new_tag] = pilot_aug_conjection(old_symbol, old_tag)
conjection_cast_table = [3, 1, 4, 2];

new_symbol = imag(old_symbol) - 1j * real(old_symbol);
new_tag = conjection_cast_table(old_tag);
end