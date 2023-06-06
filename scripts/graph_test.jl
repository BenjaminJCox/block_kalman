using LinearAlgebra
using Graphs, SimpleWeightedGraphs

using CairoMakie, GraphMakie, NetworkLayout

include(srcdir("lgssm_clustering.jl"))

A = [1 1 4 0 0 0 0 0;
    1 1 0 0 0 0 0 0;
    6 1 0 0 2 0 0 0;
    0 0 0 1 3 1 1 1;
    0 0 1 0 0 2 0 0;
    0 0 0 0 1 2 0 0;
    0 0 0 1 0 0 1 0;
    0 0 0 1 0 0 1 1]

# G = SimpleWeightedDiGraph(A)
G = DiGraph(A)

paths = floyd_warshall_shortest_paths(G)
e_paths = enumerate_paths(paths)

se = rand(UInt16)
println(se)

num_sever = 3
for n in 1:num_sever
    _, k = sever_largest_betweenness!(G)
    n_clusters = length(weakly_connected_components(G))
    @info(k)
end

f, ax, p = graphplot(G,  nlabels=repr.(1:nv(G)), nlabels_align=(:center,:center), layout = NetworkLayout.Spring(C = 10, seed = 14353), arrow_size = 20, node_size = 5 .* degree(G), curve_distance = .5)
# offsets = 0.5 * (p[:node_pos][] .- p[:node_pos][][1])
# offsets[1] = Point2f(0, 0.1)
offsets = [Point2(x,x) for x in 1 .* ones(nv(G))]
p.nlabels_offset[] = offsets
autolimits!(ax)
hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
f

G2 = DiGraph(A)
f2, ax2, p2 = graphplot(G2,  nlabels=repr.(1:nv(G2)), nlabels_align=(:center,:center), layout = NetworkLayout.Spring(C = 10, seed = 14353), arrow_size = 20, node_size = 5 .* degree(G2), curve_distance = .5)
# offsets = 0.5 * (p[:node_pos][] .- p[:node_pos][][1])
# offsets[1] = Point2f(0, 0.1)
offsets = [Point2(x,x) for x in 1 .* ones(nv(G2))]
p2.nlabels_offset[] = offsets
autolimits!(ax2)
hidedecorations!(ax2); hidespines!(ax2)
ax2.aspect = DataAspect()
f2