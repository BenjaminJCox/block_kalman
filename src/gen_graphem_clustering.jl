include("graphem_clustering.jl")

function GN_condition(G)
    _B = determine_edge_betweenness(G)
    _max = maximum(_B)
    M = zeros(Int, size(_B))
    M[_B .== _max] .= 1
    return M
end

function modprior_effect(r, M, α)
    @assert size(r) == size(M)
    r_copy = copy(r)
    r_copy[M .== 1] .*= α
    return r_copy
end

function cluster_stop(G, n_clust)
    if length(weakly_connected_components(G)) >= n_clust
        return true
    else
        return false
    end
end

function cartvector_transpose(CV)
    TCV = []
    for idx in CV
        if idx[1] != idx[2]
            push!(TCV, CartesianIndex(idx[2], idx[1]))
        end
    end
    return TCV
end

function cartvector_tupletranspose(CV)::Vector{Tuple}
    TCV = []
    for idx in CV
        if idx[1] != idx[2]
            _pv = (idx, CartesianIndex(idx[2], idx[1]))
            push!(TCV, _pv)
        else
            _pv = (idx,)
            push!(TCV, _pv)
        end
    end
    return TCV
end

function Stable_GraphEM_clustering_finputs(
    y,
    H,
    Q,
    R,
    μ₀,
    Σ₀,
    condition,
    effect,
    stop;
    pop_both = false,
    r::Matrix{Float64} = 30.0 .* collect(I(size(Q, 1))),
    init = _prox_stable(_create_adjacency_AR1(size(Q, 1), 0.1) .+ 0.1 .* randn(size(Q)), η),
    rand_reinit = true,
    inform_init = true,
    ϵ = 1e-3,
    Ei = 50,
    Mi = 1000,
    max_iters = 20,
    weighting_function = identity,
    multiple_descent = false,
)
    @info("----------------")
    @info("PM Block KF")
    @info("Initialisation")
    _state_dim = size(Q, 1)
    M = zeros(Int, size(Q))
    initial_estimate = GraphEM_stable(
        y,
        H,
        Q,
        R,
        μ₀,
        Σ₀;
        λ = 0.33,
        γ = (0.66) * 0.99,
        η = 0.99,
        r = r,
        A₀ = init,
        E_iterations = Ei,
        M_iterations = Mi,
        ϵ = ϵ,
        ξ = 1e-6,
    )
    _r_copy = copy(r)
    if inform_init
        _r_copy[initial_estimate .== 0.0] .= Inf
    end
    new_estimate = initial_estimate
    # dense_elements = findall(!=(0), initial_estimate)
    G = SimpleWeightedDiGraph(abs.(weighting_function(initial_estimate)))
    out_array = Vector{Matrix{Float64}}(undef, max_iters)
    out_array[1] = initial_estimate
    c_iters = 1
    _ll = Vector{Float64}(undef, max_iters)
    _ll[1] = _kalman(y, initial_estimate, H, Q, R, μ₀, Σ₀; drop_priors = true, likelihood = true)[3]
    stopcond = stop(G, out_array, _ll)
    @info("----------------")
    while (!stopcond) && (c_iters < max_iters)
        @info("Iteration $(c_iters)")
        M = condition(G, M)
        el = findall(M .== 1)
        if pop_both
            el = cartvector_tupletranspose(el)
            M = M + M'
            M[M .> 1] .= 1
        else
            el = [(i,) for i in el]
        end
        if multiple_descent
            @info("Comparing modification of element(s) $(Tuple(el))")
            if rand_reinit
                A0 = zeros(_state_dim, _state_dim)
                A0 = _prox_stable(_create_adjacency_AR1(size(Q, 1), 0.1) .+ 0.1 .* randn(size(Q)), η)
            else
                A0 = new_estimate
            end
            num_compare = length(el)
            likes = zeros(num_compare)
            estimates = zeros(num_compare, size(initial_estimate)...)
            rs = zeros(num_compare, size(initial_estimate)...)
            Threads.@threads for i = 1:num_compare
                _test_indices = el[i]
                # @info(el[i])
                _M_temp = zero(M)
                for idx in _test_indices
                    _M_temp[idx] = 1
                end
                _r_temp = effect(_r_copy, _M_temp)
                rs[i, :, :] = copy(_r_temp)
                estimates[i, :, :] = GraphEM_stable(
                    y,
                    H,
                    Q,
                    R,
                    μ₀,
                    Σ₀;
                    λ = 0.33,
                    γ = (0.66) * 0.99,
                    η = 0.99,
                    r = _r_temp,
                    A₀ = A0,
                    E_iterations = Ei,
                    M_iterations = Mi,
                    ϵ = ϵ,
                    ξ = 1e-6,
                )
                likes[i] = _kalman(y, estimates[i, :, :], H, Q, R, μ₀, Σ₀; drop_priors = true, likelihood = true)[3]
                @info("ll = $(likes[i]) after modifying $(el[i])")
            end
            to_remove = argmax(likes)
            @info("-------")
            @info("Modifying element $(el[to_remove])")
            @info("-------")
            new_estimate = deepcopy(estimates[to_remove, :, :])
            _r_copy = deepcopy(rs[to_remove, :, :])
        else
            @info("Modifying element(s) $(Tuple(el))")
            if rand_reinit
                # new_estimate = graphEM(dimA, steps, Y, H, m0, P, Q, R; γ = γ, dense_indices = dense_elements, θ = θ)
                A0 = _prox_stable(_create_adjacency_AR1(size(Q, 1), 0.1) .+ 0.1 .* randn(size(Q)), η)
            else
                A0 = out_array[c_iters]
            end
            _r_copy = effect(_r_copy, M)
            new_estimate = GraphEM_stable(
                y,
                H,
                Q,
                R,
                μ₀,
                Σ₀;
                λ = 0.33,
                γ = (0.66) * 0.99,
                η = 0.99,
                r = _r_copy,
                A₀ = A0,
                E_iterations = Ei,
                M_iterations = Mi,
                ϵ = ϵ,
                ξ = 1e-6,
            )
        end
        out_array[c_iters+1] = new_estimate
        _ll[c_iters+1] = _kalman(y, new_estimate, H, Q, R, μ₀, Σ₀; drop_priors = true, likelihood = true)[3]
        display(new_estimate)
        G = SimpleWeightedDiGraph(abs.(weighting_function(new_estimate)))
        stopcond = stop(G, out_array, _ll)
        n_clusters = length(weakly_connected_components(G))
        # display(_r_copy)
        @info("Now have $(n_clusters) clusters")
        c_iters += 1
        @info("----------------")
    end
    if c_iters >= max_iters
        @warn("Maximum number of iterations reached")
    end
    # return out_array[1:(c_iters-1)]
    _assigned = filter(x -> isassigned(out_array, x), 1:length(out_array))
    return (out_array[_assigned], _ll[_assigned])
end
