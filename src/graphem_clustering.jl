using LinearAlgebra
using Graphs, SimpleWeightedGraphs
using MLBase

include("graphem_prereq.jl")

import SimpleWeightedGraphs.rem_edge!
using SparseArrays

function rem_edge!(g::SimpleWeightedDiGraph, e::SimpleWeightedGraphEdge)
    has_edge(g, e) || return false
    U = weighttype(g)
    @inbounds g.weights[dst(e), src(e)] = zero(U)
    SparseArrays.dropzeros!(g.weights)
    return true
end

function rem_edge!(g::AbstractSimpleWeightedGraph, e::SimpleWeightedGraphEdge)
    has_edge(g, e) || return false
    U = weighttype(g)
    @inbounds g.weights[dst(e), src(e)] = zero(U)
    @inbounds g.weights[src(e), dst(e)] = zero(U)
    SparseArrays.dropzeros!(g.weights)
    return true
end

function determine_edge_betweenness(G)
    Gc = copy(G)
    if hasproperty(Gc, :weights)
        Gc.weights = abs.(Gc.weights)
    end
    paths = floyd_warshall_shortest_paths(Gc)
    e_paths = enumerate_paths(paths)
    # @info("e_paths exists")
    betweenness_matrix = zeros(Int, nv(G), nv(G))
    for v_paths in e_paths
        for path in v_paths
            path_length = length(path)
            if path_length > 0
                for v = 2:path_length
                    s = path[v-1]
                    d = path[v]
                    betweenness_matrix[s, d] += 1
                end
            end
        end
    end
    return betweenness_matrix
end

function sever_largest_betweenness!(G; pop_multiple = true)
    betweenness_matrix = determine_edge_betweenness(G)
    # _am = argmax(betweenness_matrix)
    _lb = maximum(betweenness_matrix)
    @info("Largest Betweenness: $_lb")
    _amv = findall(e -> e == _lb, betweenness_matrix)
    if pop_multiple
        for _am in _amv
            rem_edge!(G, _am[1], _am[2])
        # if has_edge(G, _am[2], _am[1])
            # rem_edge!(G, _am[2], _am[1])
        # end
        end
        return (G, _amv)
    else
        rem_edge!(G, _amv[1][1], _amv[1][2])
        return (G, [_amv[1]])
    end
end

function pop_cart_from_edges!(src, dest, cart_list; both = false)
    filter!(x -> x != CartesianIndex(src, dest), cart_list)
    if both
        filter!(x -> x != CartesianIndex(dest, src), cart_list)
    end
    return cart_list
end

function Stable_GraphEM_clustering(
    num_clusters,
    y,
    H,
    Q,
    R,
    μ₀,
    Σ₀;
    directed = true,
    pop_both = false,
    pop_multiple = true,
    r = 20.0,
    init = _prox_stable(_create_adjacency_AR1(size(Q, 1), 0.1) .+ 0.1 .* randn(size(Q)), η),
    rand_reinit = true,
    ϵ = 1e-3,
    Ei = 50,
    Mi = 1000,
    max_iters = 20,
    weighting_function = identity,
    multiple_descent = false
)
    @info("----------------")
    @info("Block KF")
    @info("Initialisation")
    _state_dim = size(Q, 1)
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
        dense_indices = eachindex(init),
    )
    new_estimate = initial_estimate
    dense_elements = findall(!=(0), initial_estimate)
    if directed
        G = SimpleWeightedDiGraph(abs.(weighting_function(initial_estimate)))
    else
        G = Graph(abs.(weighting_function(initial_estimate)) .+ abs.(weighting_function(initial_estimate))')
        pop_both = true
    end
    out_array = Vector{Matrix{Float64}}(undef, max_iters)
    out_array[1] = initial_estimate
    c_iters = 1
    n_clusters = length(weakly_connected_components(G))
    _ll = Vector{Float64}(undef, max_iters)
    _ll[1] = _kalman(y, initial_estimate, H, Q, R, μ₀, Σ₀; drop_priors = true, likelihood = true)[3]
    @info("----------------")
    while (n_clusters < num_clusters) && (c_iters < max_iters)
        @info("Iteration $(c_iters)")
        # perform pseudo girvan newman clustering
        G, el = sever_largest_betweenness!(G, pop_multiple = pop_multiple)
        if multiple_descent && pop_multiple
            @info("Comparing removal of element(s) $(Tuple(el))")
            if rand_reinit
                # new_estimate = graphEM(dimA, steps, Y, H, m0, P, Q, R; γ = γ, dense_indices = dense_elements, θ = θ)
                A0 = zeros(_state_dim, _state_dim)
                A0[dense_elements] = _prox_stable(_create_adjacency_AR1(size(Q, 1), 0.1) .+ 0.1 .* randn(size(Q)), η)[dense_elements]
            else
                # new_estimate = graphEM(dimA, steps, Y, H, m0, P, Q, R; γ = γ, dense_indices = dense_elements, θ = θ, init = new_estimate[dense_elements])
                A0 = zeros(_state_dim, _state_dim)
                A0[dense_elements] = new_estimate[dense_elements]
            end
            num_compare = length(el)
            likes = zeros(num_compare)
            estimates = zeros(num_compare, size(initial_estimate)...)
            Threads.@threads for i in 1:num_compare
                _cidxs = deepcopy(dense_elements)
                pop_cart_from_edges!(el[i][1], el[i][2], _cidxs, both = pop_both)
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
                    r = r,
                    A₀ = A0,
                    E_iterations = Ei,
                    M_iterations = Mi,
                    ϵ = ϵ,
                    ξ = 1e-6,
                    dense_indices = _cidxs,
                )
                likes[i] = _kalman(y, estimates[i, :, :], H, Q, R, μ₀, Σ₀; drop_priors = true, likelihood = true)[3]
            end
            to_remove = argmax(likes)
            pop_cart_from_edges!(el[to_remove][1], el[to_remove][2], dense_elements, both = pop_both)
            new_estimate = deepcopy(estimates[to_remove, :, :])
        else
            @info("Removing element(s) $(Tuple(el))")
            for _el in el
                pop_cart_from_edges!(_el[1], _el[2], dense_elements, both = pop_both)
            end
            if rand_reinit
                # new_estimate = graphEM(dimA, steps, Y, H, m0, P, Q, R; γ = γ, dense_indices = dense_elements, θ = θ)
                A0 = zeros(_state_dim, _state_dim)
                A0[dense_elements] = _prox_stable(_create_adjacency_AR1(size(Q, 1), 0.1) .+ 0.1 .* randn(size(Q)), η)[dense_elements]
            else
                # new_estimate = graphEM(dimA, steps, Y, H, m0, P, Q, R; γ = γ, dense_indices = dense_elements, θ = θ, init = new_estimate[dense_elements])
                A0 = zeros(_state_dim, _state_dim)
                A0[dense_elements] = new_estimate[dense_elements]
            end
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
                r = r,
                A₀ = A0,
                E_iterations = Ei,
                M_iterations = Mi,
                ϵ = ϵ,
                ξ = 1e-6,
                dense_indices = dense_elements,
            )
        end
        out_array[c_iters+1] = new_estimate
        _ll[c_iters+1] = _kalman(y, new_estimate, H, Q, R, μ₀, Σ₀; drop_priors = true, likelihood = true)[3]
        if directed
            G = SimpleWeightedDiGraph(abs.(weighting_function(new_estimate)))
        else
            G = Graph(abs.(weighting_function(new_estimate)) .+ abs.(weighting_function(new_estimate))')
        end
        n_clusters = length(weakly_connected_components(G))
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

function matrix_rmse(est_matrix, true_matrix)
    E = true_matrix .- est_matrix
    SQE = E.^2
    MSE = mean(SQE)
    RMSE = sqrt(MSE)
    return RMSE
end

function prec_rec_graphem(true_A, gem_A)
    true_sparse = (true_A .== 0.0)
    est_sparse = (abs.(gem_A) .== 0.0)
    ts_vec = vec(true_sparse)
    es_vec = vec(est_sparse)
    eroc = roc(ts_vec, es_vec)
    prec = precision(eroc)
    rec = recall(eroc)
    f1 = f1score(eroc)
    spec = true_negative_rate(eroc)
    rmse = matrix_rmse(gem_A, true_A)
    return @dict prec rec f1 spec rmse
end
