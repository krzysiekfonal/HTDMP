function Xht = fht_decomposition( X, R, alg, varargin )

L = size(R,2);
Xht = ft_decomposition(X, cell2mat(R{1}), alg, varargin{1:end});
T = hnt_decomposition_internal(Xht{ndims(X)}, R, L, alg, varargin{1:end});
Xht(end) = [];
Xht = {Xht, T{:}};
Xht = vertcat(Xht{:});

end


function Xht = hnt_decomposition_internal( G, R, L, alg, varargin )
%HNT_DECOMPOSITION Summary of this function goes here
%   Detailed explanation goes here

Xht = cell(L,1);
dims = size(G);
persisten_mode_size = dims(end);
%R = {num2cell(dims), R{1:end}}; % add 0-level in R-tree (because matlab index from 1
                 % levels need to be shifted, so 2 means original 1, 1
                 % means original 0 i.e. ranks from input tensor, not core
                 % tensor

for l=1:L-1
    Xht{l} = cell(2^(L-l),1);
    for nl = 1:2^(L-l)
        W = pair_mtx(G, nl);
        [A, B] = lrmf(W, alg, R{l+1}{nl}, varargin);
        Xht{l}{nl} = reshape(A, R{l}{2*nl-1}, R{l}{2*nl}, R{l+1}{nl});
        G = reshape(B, ...
            [R{l+1}{nl} R{l+1}{1:nl-1} R{l}{(2*nl+1):(2^(L-l+1))}...
            persisten_mode_size]);
        p = [2:nl 1 nl+1:ndims(G)];
        G = permute(G, p);
    end
end

Xht{L} = cell(1,1);
Xht{L}{1} = G;

end

