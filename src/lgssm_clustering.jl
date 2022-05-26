using LinearAlgebra
using Graphs, SimpleWeightedGraphs

include("linear_filter_smoother.jl")

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
    paths = floyd_warshall_shortest_paths(G)
    e_paths = enumerate_paths(paths)
    betweenness_matrix = zeros(Int, nv(G), nv(G))
    for v_paths in e_paths
        for path in v_paths
            path_length = length(path)
            if path_length > 0
                for v in 2:path_length
                    s = path[v-1]
                    d = path[v]
                    betweenness_matrix[s,d] += 1
                end
            end
        end
    end
    return betweenness_matrix
end

function sever_largest_betweenness!(G)
    betweenness_matrix = determine_edge_betweenness(G)
    _am = argmax(betweenness_matrix)
    rem_edge!(G, _am[1], _am[2])
    if has_edge(G, _am[2], _am[1])
        rem_edge!(G, _am[2], _am[1])
    end
    return (G, _am)
end

function pop_cart_from_edges!(src, dest, cart_list; both = true)
    filter!(x -> x != CartesianIndex(src, dest), cart_list)
    if both
        filter!(x -> x != CartesianIndex(dest, src), cart_list)
    end
    return cart_list
end

function graphem_clustering(num_clusters, dimA, steps, Y, H, m0, P, Q, R; γ = 0.5, weighted = false, max_iters = 20)
    initial_estimate = graphEM(dimA, steps, Y, H, m0, P, Q, R; γ = γ)
    dense_elements = findall(!=(0), initial_estimate)
    if weighted
        G = SimpleWeightedDiGraph(initial_estimate)
    else
        G = DiGraph(initial_estimate)
    end
    out_array = Vector{Matrix{Float64}}(undef, max_iters)
    out_array[1] = initial_estimate
    c_iters = 2
    n_clusters = length(weakly_connected_components(G))
    _ll = Vector{Float64}(undef, max_iters)
    _ll[1] = _perform_kalman(Y, initial_estimate, H, m0, P, Q, R; lle = true)[3]
    while (n_clusters < num_clusters) && (c_iters < max_iters)
        G, el = sever_largest_betweenness!(G)
        @info("Removing element $(el)")
        pop_cart_from_edges!(el[1], el[2], dense_elements)
        new_estimate = graphEM(dimA, steps, Y, H, m0, P, Q, R; γ = γ, dense_indices = dense_elements)
        out_array[c_iters] = new_estimate
        _ll[c_iters] = _perform_kalman(Y, new_estimate, H, m0, P, Q, R; lle = true)[3]
        if weighted
            G = SimpleWeightedDiGraph(new_estimate)
        else
            G = DiGraph(new_estimate)
        end
        n_clusters = length(weakly_connected_components(G))
        @info("Now have $(n_clusters) clusters")
        c_iters += 1
    end
    # return out_array[1:(c_iters-1)]
    _assigned = filter(x -> isassigned(out_array, x), 1:length(out_array))
    return (out_array[_assigned], _ll[_assigned])
end
