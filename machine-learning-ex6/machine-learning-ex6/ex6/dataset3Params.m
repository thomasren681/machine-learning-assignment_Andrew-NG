function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

valset = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%define a set to try different values of C and sigma
m = length(valset);
costs = zeros(m, m);
%store the costs of each value we try in the costs matrix
for i = 1:m
  for j = 1:m
    C_temp = valset(i);
    sigma_temp = valset(j);
    model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
    p = svmPredict(model, Xval);
    costs(i, j) = mean(double(p ~= yval));
    %compute the wrong predictions and store it in the matrix
  endfor
endfor
min_cost = min(min(costs));
[i,j] = find(min_cost == costs);
C = valset(i);
sigma = valset(j);



% =========================================================================

end
