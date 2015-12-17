function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%X=[ones(m,1) X];
%theta=[0; theta];
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h=sigmoid(X*theta);
grad(1)=(1/m)*((h-y)'*X(:,1));
for i=2:size(grad,1)
grad(i)=(1/m)*((h-y)'*X(:,i))+(lambda/m)*theta(i);
end

term1=((-1)*y)'*log(h);
term2=(ones(size(y))-y)'*(log(1-h));
term3=(lambda/(2*m))*sum(theta(2:size(theta)).^2);
J=(1/m)* (term1-term2) +term3;





% =============================================================

end
