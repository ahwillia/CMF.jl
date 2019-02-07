module ProjALS

# Imports
include("./common.jl")


function update(data, W, H, meta, options)
    if (meta == nothing)
        meta = ProjALSMeta(data, W, H)
    end

    # Debug
    println(norm(W), " : ", norm(H))

    _update_W!(data, W, H)
    _update_H!(data, W, H, meta)

    return compute_loss(data, W, H), meta
end


mutable struct ProjALSMeta
    shift_matrices

    # Initialization
    function ProjALSMeta(data, W, H)
        L = size(W, 1)
        T = size(H, 2)
        
        shift_matrices = []

        for l = 1:L
            push!(shift_matrices, shift_matrix(T, -l+1))
        end

        return new(shift_matrices)
    end
end


function _update_W!(data, W, H)
    L, N, K = size(W)

    # Build the larger H matrix
    H_stacked = shift_and_stack(H, L)

    # Solve via projected least squares
    W_stacked_trans = max.(0, H_stacked' \ data')

    # Update W
    for l = 0:(L-1)
        W[l+1, :, :] = W_stacked_trans[K*l+1:K*(l+1), :]'
    end
end


function _update_H!(data, W, H, meta)
    L, N, K = size(W)
    K, T = size(H)
    
    # Construct matrix F
    F = kron(meta.shift_matrices[1], W[1, :, :])
    for l = 2:L
        F .+= kron(meta.shift_matrices[l], W[l, :, :])
    end

    # Solve via projected least squares
    H_vec = max.(0, F \ reshape(data, N*T))
    
    # Reshape back into H
    H[:, :] = reshape(H_vec, K, T)
end


end  # module
;
