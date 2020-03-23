function M = pair_mtx( X, n )
%PAIR_MTX Matricization, which unfold X tensor having n and n+1 mode as the rows
% and rest as a columns

dims = size(X);
N = ndims(X);

A = [n n+1]; % set of modes as a rows
B = setdiff(1:N, A); % set of modes as a columns

M = reshape(permute(X, [A B]), prod(dims(A)), prod(dims(B)));

end

