function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
a2 = sigmoid([ones(m, 1) X] * Theta1');
a3 = sigmoid([ones(m, 1) a2] * Theta2');
for i=1:m,
	
	yi=zeros(num_labels,1);
	if y(i)==0 
		yi(10)=1;
	else
		yi(y(i))=1;
	end,
	J=J+(-log(a3(i,:))*yi-(1-yi)'*log(1-(a3(i,:))'));
end,

J=J*(1/m);
	
theta1=Theta1(:,2:end);
theta2=Theta2(:,2:end);

for i=1:size(theta1,1)
	for j=1:size(theta1,2)
		theta1(i,j)=theta1(i,j)*theta1(i,j);
	end,
end,

for i=1:size(theta2,1)
	for j=1:size(theta2,2)
		theta2(i,j)=theta2(i,j)*theta2(i,j);
	end,
end,

sum1=sum(sum(theta1,2),1);
sum2=sum(sum(theta2,2),1);
a=(lambda*(sum1+sum2))/(2*m);
J=J+a;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
a1=[ones(m, 1) X];
z2=a1 * Theta1';
a2=sigmoid(z2);
a2=[ones(m, 1) a2];
z3=a2 * Theta2';
a3=sigmoid(z3);


for t=1:m,
	a_3=a3(t,:)';
	a_1=a1(t,:);
	a_2=a2(t,:);
	yt=zeros(num_labels,1);
	yt(y(t))=1;
	delta_3=a_3-yt;
	z_2=z2(t,:);
	
	sg=(sigmoidGradient(z_2))';
	sg=[1; sg];  %26x1
	temp=Theta2'*delta_3;	
	delta_2=temp.*sg;
	delta_2=delta_2(2:end);
	
	
	incl1=delta_2*a_1;
	Theta1_grad = Theta1_grad+ incl1;
	
	incl2=delta_3*a_2;
	Theta2_grad = Theta2_grad +incl2;
end,

Theta1_grad= Theta1_grad*(1/m);
Theta2_grad= Theta2_grad*(1/m);




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
for i=1:size(Theta1_grad,1)
	for j=2:size(Theta1_grad,2)
		temp=Theta1(i,j)*(lambda/m);
		Theta1_grad(i,j)=Theta1_grad(i,j)+temp;
	end;
end;

for i=1:size(Theta2_grad,1)
	for j=2:size(Theta2_grad,2)
		temp=Theta2(i,j)*(lambda/m);
		Theta2_grad(i,j)=Theta2_grad(i,j)+ temp;
	end;
end;


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
