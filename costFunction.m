function [J, grad] = costFunction(theta, X, y)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

J = (-y' * log(sigmoid(X*theta)) - (ones(size(y)) - y)' * log(ones(size(y)) - (sigmoid(X*theta))))/m;

grad = (X' * (sigmoid(X*theta) - y))/m;

end
