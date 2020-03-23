function [ Xtt ] = ft_decomposition( X, ranks, alg, varargin)

dim = size(X);
N = ndims(X);
Xtt = cell(N, 1);

% compute matrices of leaves
for n = 1:N-1
    W = reshape(permute(double(X),[n [1:n-1,n+1:N]]),dim(n), prod(dim)/dim(n));
    [Xtt{n}, B] = lrmf(W, alg, ranks(n), varargin);
    X = reshape(B, [ranks(n:-1:1) dim(n+1:N)]);
    dim(n) = ranks(n);
end
Xtt{N} = permute(X,[N-1:-1:1 N]);

end

