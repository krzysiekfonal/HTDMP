function [G] = generate_graph(tensor, k_neighbors, t)
use_gpu = false;
elements = size(tensor);
G = zeros(elements(3), elements(3));
distances = zeros(elements(3), elements(3));
if use_gpu
    tensor = gpuArray(tensor);
    distances = gpuArray(distances);
    G = gpuArray(G);
end
if ~isscalar(k_neighbors)
    for i = 1:elements(3)
        if (i < 10 || ~mod(i,10))
            disp(['calculated distances for: ',num2str(i)]);
        end
        for j = i:elements(3)
            if(k_neighbors(i) == k_neighbors(j))
                G(i, j) = exp(-(norm((tensor(:,:,i) - tensor(:,:,j)), 'fro').^2)/t);
                G(j, i) = G(i, j);
            end
        end
    end
else
    for i = 1:elements(3)
        if (i < 10 || ~mod(i,10))
            disp(['calculated distances for: ',num2str(i)]);
        end
        %for j = (i+1):elements(3)
        for j = (i):elements(3)
            distances(i, j) = norm((tensor(:,:,i) - tensor(:,:,j)), 'fro');
            distances(j, i) = distances(i, j);
        end
        [~, indexes] = sort(distances(i,:));
        %indexes = indexes(2:k_neighbors+1);
        indexes = indexes(1:k_neighbors+1);
        G(i, indexes) = exp(-(distances(i, indexes).^2)/t);
    end
end
if use_gpu
    G = gather(G);
end
G = sparse(G);
end