function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


%X = [ones(5000,1) X];
%a2 = sigmoid(X*(Theta1'));%a2 has size of 5000x25
%a2 = [ones(5000,1) a2];%append the bias column to the a2 matrix
%a3 = sigmoid(a2*(Theta2'));%a3 has the size of 5000x10
%now we pick the most possible prediction for each picture...
%i.e. each row that has being unrolled from a 20x20 pixel picture
%[maxx p] = max(a3, [], 2);
X = [ones(m, 1) X];
a2 = [ones(m, 1) sigmoid(X * Theta1')];
%size(a2)
[maxx, p] = max(sigmoid(a2 * Theta2'), [], 2);



% =========================================================================


end
