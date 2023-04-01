using DrWatson
using Distributions, Random, LinearAlgebra, BlockDiagonals
using InvertedIndices
using CairoMakie
using GraphMakie, NetworkLayout
using BenchmarkTools
# using MAT

include(srcdir("linear_filter_smoother.jl"))
include(srcdir("graphem_prereq.jl"))
include(srcdir("graphem_clustering.jl"))
include(srcdir("rw_tools.jl"))


Random.seed!(0xabcdefabcdef)


_blocksize = 2
# _Mn = MatrixNormal(zeros(_blocksize, _blocksize), Matrix(1.0 * I(_blocksize)), Matrix(1.0 * I(_blocksize)))
# A_blocks = [_create_adjacency_AR1(_blocksize, 0.1) for _ = 1:8]
A_blocks = [rand(_blocksize, _blocksize) .+ 2.0 for _ = 1:4]
A = BlockDiagonal(A_blocks)
A = Matrix(A)
a_dim = size(A,1)
A ./= 1 .* eigmax(A)

qr_mult = 1.0
Q = Matrix(qr_mult .* I(a_dim))
# Q .+= 0.000001
# R = Matrix(qr_mult .* I(a_dim))
# Q = Matrix(1e-2 .* I(a_dim))
R = Matrix(1e-2 .* I(a_dim))
H = Matrix(1.0 .* I(a_dim))
# R = Matrix(qr_mult .* I(4))
# H = 1.0 .* [1 1 0 0 0 0 0 0; 0 0 1 1 0 0 0 0; 0 0 0 0 1 1 0 0; 0 0 0 0 0 0 1 1]

o_dim = size(H, 1)


P = Matrix(1e-8 .* I(a_dim))

m0 = ones(a_dim)

T = 100

X = zeros(a_dim, T)
Y = zeros(o_dim, T)

prs_noise = MvNormal(Q)
obs_noise = MvNormal(R)
prior_state = MvNormal(m0, P)

for t = 1:T
    if t == 1
        X[:, 1] = A * rand(prior_state) .+ rand(prs_noise)
        Y[:, 1] = H * X[:, 1] .+ rand(obs_noise)
    else
        X[:, t] = A * X[:, t-1] .+ rand(prs_noise)
        Y[:, t] = H * X[:, t] .+ rand(obs_noise)
    end
end

okalm_n = _kalman(Y, A, H, Q, R, m0, P, likelihood = true)
okalm_n_s = _static_kalman(SMatrix{o_dim, T}(Y), SMatrix{a_dim, a_dim}(A), SMatrix{o_dim, a_dim}(H), SMatrix{a_dim, a_dim}(Q), SMatrix{o_dim, o_dim}(R), SVector{a_dim}(m0),  SMatrix{a_dim, a_dim}(P), likelihood = true)
@profiler for i = 1:100
    okalm_n = _kalman(Y, A, H, Q, R, m0, P, likelihood = true)
end
@profiler for i = 1:10000
    okalm_n_s = _static_kalman(SMatrix{o_dim, T}(Y), SMatrix{a_dim, a_dim}(A), SMatrix{o_dim, a_dim}(H), SMatrix{a_dim, a_dim}(Q), SMatrix{o_dim, o_dim}(R), SVector{a_dim}(m0),  SMatrix{a_dim, a_dim}(P), likelihood = true)
end

@info("DYNAMIC")
@btime okalm_n = _kalman(Y, A, H, Q, R, m0, P, likelihood = true)
@info("NAIVE STATIC")
@btime okalm_n_s = _static_kalman(SMatrix{o_dim, T}(Y), SMatrix{a_dim, a_dim}(A), SMatrix{o_dim, a_dim}(H), SMatrix{a_dim, a_dim}(Q), SMatrix{o_dim, o_dim}(R), SVector{a_dim}(m0),  SMatrix{a_dim, a_dim}(P), likelihood = true)

# γ = exp(0.25)
γ = 0.1
η = 0.01

r = 35

# A_init = MLEM_A(a_dim, 25, Y, H, m0, P, Q, R)
A_init = rand(a_dim, a_dim)
A_init = (A_init + A_init') / 2

a_gem1 = GraphEM_stable(Y, H, Q, R, m0, P, r = r, λ = 0.99 / 3)

display(a_gem1)

# _R = zeros(a_dim, a_dim)
# _R .+= r
# _R[diagind(_R)] .*= 0.8
# a_gem2 = GraphEM_stable(Y, H, Q, R, m0, P, r = _R, λ = 0.99 / 3)
# display(a_gem2)
# @btime okalm_o = _perform_kalman(Y, A, H, m0, P, Q, R, lle = true)


# a_gem = graphEM(a_dim, 50, Y, H, m0, P, Q, R, γ = γ, θ = 1.0, init = vec(A_init)); display(a_gem)
# a_gem_clstr = graphem_clustering(4, a_dim, 50, Y, H, m0, P, Q, R, γ = γ, θ = 1.0, directed = true, max_iters = 50, init = vec(A_init), rand_reinit = true)

quadweight(x) = x .^ 2
function gmw_e(x, μ, σ)
    if x != 0
        x = 1 - exp(-0.5*(x/σ)^2)/σ
    else
        x = x
    end
end
function pmw_e(x, γ)
    if x != 0
        x = x .^ γ
    end
end
function pmw_m(x)
    rv = pmw_e(x, 2)
    rvext = extrema(rv)
    rv = (rv .- rvext[1])./(rvext[2] - rvext[1])
    return rv
end

qmw(x) = qmw_e.(x, mean(x), 1)
a_gem_clstr = Stable_GraphEM_clustering(4, Y, H, Q, R, m0, P, rand_reinit = true, r = r, directed = true, pop_multiple = true, weighting_function = identity, multiple_descent = true)

true_filtered = _perform_kalman(Y, A, H, m0, P, Q, R)
# true_filtered = _kalman(Y, A, H, Q, R, m0, P)
reduction_filtered = _perform_kalman(Y, a_gem_clstr[1][end], H, m0, P, Q, R)
gem_filtered = _perform_kalman(Y, a_gem1, H, m0, P, Q, R)

sq_error_true = cumsum(sqrt.(vec(sum((true_filtered[1] .- X) .^ 2, dims = 1))))
sq_error_redu = cumsum(sqrt.(vec(sum((reduction_filtered[1] .- X) .^ 2, dims = 1))))
sq_error_gem = cumsum(sqrt.(vec(sum((gem_filtered[1] .- X) .^ 2, dims = 1))))

dtr_y = (sq_error_redu .+ sq_error_true .+ sq_error_gem) ./ 3
dtr_x = hcat(repeat([1], T), collect(1:T))
dtr_β = dtr_x \ dtr_y
dtr_fitted = dtr_x * dtr_β



f = Figure(resolution = (1600, 800))
ax_true = Axis(f[1, 1], title = "Cumulative RtSq. Error")
scatterlines!(1:T, sq_error_true, label = "True Filter")
scatterlines!(1:T, sq_error_redu, label = "Reduced Filter")
scatterlines!(1:T, sq_error_gem, label = "GraphEM Filter")
lines!(1:T, dtr_fitted, label = "Trend", color = "red")
xlims!(1, T)
axislegend(position = :rb)
f

f1 = Figure(resolution = (1600, 800))
ax_true = Axis(f1[1, 1], title = "Detrended Cumulative RtSq. Error")
lines!(1:T, sq_error_true .- dtr_fitted, label = "True Filter")
lines!(1:T, sq_error_redu .- dtr_fitted, label = "Reduced Filter")
lines!(1:T, sq_error_gem .- dtr_fitted, label = "GraphEM Filter")
xlims!(1, T)
axislegend(position = :lb)
f1

f2 = Figure(resolution = (1600, max(1600, 200 * a_dim)))
for dimension = 1:a_dim
    ax_tempo = Axis(f2[dimension, 1])
    scatter!(ax_tempo, 1:T, X[dimension, :], label = "True State", color = "mediumpurple1")
    # lines!(ax_temp, 1:T, true_filtered[1][dimension,:], label = "True Filter")
    lines!(ax_tempo, 1:T, reduction_filtered[1][dimension, :], label = "Block KF")
    lines!(ax_tempo, 1:T, gem_filtered[1][dimension, :], label = "GraphEM")
    xlims!(0, T + 1)
    if dimension == a_dim
        Legend(f2[1:a_dim, 2], ax_tempo)
    end
end
rowgap!(f2.layout, 1)

sum((a_gem_clstr[1][end] .- A) .^ 2)
sum((a_gem1 .- A) .^ 2)
# sum((A_init .- A).^2)

G = SimpleWeightedDiGraph(A)
_GTG = SimpleWeightedDiGraph(A)

function colour_edges(truth, est, truecol, falsecol)
    n = 1
    _ec_itr = [falsecol for _ in 1:ne(est)]
    for _e in edges(est)
        if has_edge(truth, src(_e), dst(_e))
            _ec_itr[n] = truecol
        end
        n += 1
    end
    return _ec_itr
end

wm = Matrix(Graphs.weights(G))
wv = vec(wm[wm.!=0.0])

set_theme!(resolution = (200, 200))

_ec = :turquoise3
_nc = :lightblue3

_true_e = edges(G)

# _nwl = NetworkLayout.Spring(C = 6, seed = 14353)
_nwl = NetworkLayout.SFDP(seed = 14353, C = 0.2, K = 1)

f4, ax4, p4 = graphplot(
    G,
    nlabels = repr.(1:nv(G)),
    nlabels_align = (:center, :center),
    layout = _nwl,
    arrow_size = 20,
    edge_width = 6 .* wv,
    node_size = 5 .* degree(G),
    curve_distance = 0.5,
    edge_color = _ec,
    node_color = _nc,
    selfedge_size = 80
)
# offsets = 0.5 * (p[:node_pos][] .- p[:node_pos][][1])
# offsets[1] = Point2f(0, 0.1)
offsets = [Point2(x, x) for x in 1 .* ones(nv(G))]
# p4.nlabels_offset[] = offsets
autolimits!(ax4)
hidedecorations!(ax4);
hidespines!(ax4);
# ax4.aspect = DataAspect()

f4

G = SimpleWeightedDiGraph(a_gem1)

wm = Matrix(Graphs.weights(G))
wv = vec(wm[wm.!=0.0])

_ec_gem = colour_edges(_GTG, G, _ec, :red)

f5, ax5, p5 = graphplot(
    G,
    nlabels = repr.(1:nv(G)),
    nlabels_align = (:center, :center),
    layout = _nwl,
    arrow_size = 20,
    edge_width = 6 .* wv,
    node_size = 5 .* degree(G),
    curve_distance = 0.5,
    edge_color = _ec_gem,
    node_color = _nc,
    selfedge_size = 80
)
# offsets = 0.5 * (p[:node_pos][] .- p[:node_pos][][1])
# offsets[1] = Point2f(0, 0.1)
offsets = [Point2(x, x) for x in 1 .* ones(nv(G))]
# p5.nlabels_offset[] = offsets
autolimits!(ax5)
hidedecorations!(ax5);
hidespines!(ax5);
# ax5.aspect = DataAspect()

f5

G = SimpleWeightedDiGraph(a_gem_clstr[1][end])

_ec_cdkf = colour_edges(_GTG, G, _ec, :red)

wm = Matrix(Graphs.weights(G))
wv = vec(wm[wm.!=0.0])

f6, ax6, p6 = graphplot(
    G,
    nlabels = repr.(1:nv(G)),
    nlabels_align = (:center, :center),
    layout = _nwl,
    arrow_size = 20,
    edge_width = 6 .* wv,
    node_size = 5 .* degree(G),
    curve_distance = 0.5,
    edge_color = _ec_cdkf,
    node_color = _nc,
    selfedge_size = 80
)
# offsets = 0.5 * (p[:node_pos][] .- p[:node_pos][][1])
# offsets[1] = Point2f(0, 0.1)
offsets = [Point2(x, x) for x in 1 .* ones(nv(G))]
# p6.nlabels_offset[] = offsets
autolimits!(ax6)
hidedecorations!(ax6);
hidespines!(ax6);
# ax6.aspect = DataAspect()

f6

f7 = Figure(resolution = (800, 400))
ax_temp = Axis(f7[1, 1])
scatter!(ax_temp, 1:T, X[4, :], label = "True State", color = "mediumpurple1")
lines!(ax_temp, 1:T, reduction_filtered[1][4, :], label = "Block KF")
lines!(ax_temp, 1:T, gem_filtered[1][4, :], label = "GraphEM")
xlims!(0, T + 1)
axislegend(ax_temp, merge = true, position = :lb)
# Legend(f7[2, 1], ax_temp)
rowgap!(f7.layout, 1)
f7

G = SimpleWeightedDiGraph(a_gem_clstr[1][begin])

_ec_cdkf = colour_edges(_GTG, G, _ec, :red)

wm = Matrix(Graphs.weights(G))
wv = vec(wm[wm.!=0.0])

f8, ax8, p8 = graphplot(
    G,
    nlabels = repr.(1:nv(G)),
    nlabels_align = (:center, :center),
    layout = _nwl,
    arrow_size = 20,
    edge_width = 6 .* wv,
    node_size = 5 .* degree(G),
    curve_distance = 0.5,
    edge_color = _ec_cdkf,
    node_color = _nc,
    selfedge_size = 80
)

offsets = [Point2(x, x) for x in 1 .* ones(nv(G))]
# p6.nlabels_offset[] = offsets
autolimits!(ax8)
hidedecorations!(ax8);
hidespines!(ax8);


f
f1
f2

f4
f5
f6

f8
f7

# save(plotsdir("states.pdf"), f7)
# save(plotsdir("gt_graph.pdf"), f4)
# save(plotsdir("gem_graph.pdf"), f5)
# save(plotsdir("bkf_graph.pdf"), f6)
base_gem_stats = prec_rec_graphem(A, a_gem1)
clst_gem_stats = prec_rec_graphem(A, a_gem_clstr[1][end])

println("-----------")
println("Base GraphEM")
display(base_gem_stats)

println("-----------")
println("CDKF")
display(clst_gem_stats)

lines(a_gem_clstr[2])

f6
