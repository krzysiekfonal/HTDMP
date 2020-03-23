function X = fht_projection( X, Xht, R)

L = size(R,2);
X = ft_projection_internal(X, Xht, cell2mat(R{1}));
X = fht_projection_internal(X,Xht, R, L);

end


function X = ft_projection_internal( X, Xht, ranks)

dim = size(X);
N = ndims(X);

% compute matrices of leaves
for n = 1:N-1
    W = reshape(permute(double(X),[n [1:n-1,n+1:N]]),dim(n), prod(dim)/dim(n));
    B = Xht{n}' * W;
    X = reshape(B, [ranks(n:-1:1) dim(n+1:N)]);
    dim(n) = ranks(n);
end
X = permute(X,[N-1:-1:1 N]);
end

function G = fht_projection_internal( G, Xht, R, L )
%HNT_DECOMPOSITION Summary of this function goes here
%   Detailed explanation goes here

dims = size(G);
persisten_mode_size = dims(end);
%R = {num2cell(dims), R{1:end}}; % add 0-level in R-tree (because matlab index from 1
                 % levels need to be shifted, so 2 means original 1, 1
                 % means original 0 i.e. ranks from input tensor, not core
                 % tensor

idx = ndims(G);
for l=1:L-1
    for nl = 1:2^(L-l)
        W = pair_mtx(G, nl);
        A = reshape(Xht{idx}, R{l}{2*nl-1} * R{l}{2*nl}, R{l+1}{nl});
        B = A' * W;
        G = reshape(B, ...
            [R{l+1}{nl} R{l+1}{1:nl-1} R{l}{(2*nl+1):(2^(L-l+1))}...
            persisten_mode_size]);
        p = [2:nl 1 nl+1:ndims(G)];
        G = permute(G, p);
        idx = idx+1;
    end
end

end

