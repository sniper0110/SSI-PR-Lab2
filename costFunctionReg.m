function [J, grad] = costFunctionReg(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

s = size(theta, 1);
J = (-y' * log(sigmoid(X*theta)) - (ones(size(y)) - y)' * log(ones(size(y)) - (sigmoid(X*theta))))/m + lambda * (theta(2:s,:)' * theta(2:s,:))/(2*m);              

grad(1) = ((sigmoid(X*theta) - y)' * X(:,1))/m;

for j=2:s
    grad(j) = ((sigmoid(X*theta) - y)' * X(:,j))/m + lambda/m * theta(j);
end

end
