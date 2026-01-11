manifold_file='loco';
matrice_matlab = csvread(strcat(manifold_file,'.csv'));
groups = csvread('loco_labels.csv');


x = matrice_matlab(:, 1);
y = matrice_matlab(:, 2);
z = matrice_matlab(:, 3);

figure;
scatter3(x, y, z, 15, groups, 'filled');
title(manifold_file);
xlim([-5 5]);
ylim([-5 5]);
zlim([-5 5]);
%colorbar;

