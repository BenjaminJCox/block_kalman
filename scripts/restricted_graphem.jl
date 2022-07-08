using DrWatson
using Distributions, Random, LinearAlgebra, BlockDiagonals
using InvertedIndices
using CairoMakie
# using MAT

include(srcdir("linear_filter_smoother.jl"))
include(srcdir("graphem_prereq.jl"))
include(srcdir("graphem_clustering.jl"))
include(srcdir("lgssm_clustering.jl"))


Random.seed!(0xabcdefabcf)

A_blocks = [rand(2, 2) .+ 2.0 for _ = 1:4]
a_dim = 8
A = BlockDiagonal(A_blocks)
A = Matrix(A)
A ./= 1 .* eigmax(A)

qr_mult = 1
Q = Matrix(qr_mult .* I(a_dim))
R = Matrix(qr_mult .* I(a_dim))
# Q = Matrix(1e-2 .* I(a_dim))
# R = Matrix(1e-2 .* I(a_dim))
H = Matrix(1.0 .* I(a_dim))
P = Matrix(1e-8 .* I(a_dim))

m0 = ones(a_dim)

T = 1000

X = zeros(a_dim, T)
Y = zeros(a_dim, T)

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

# matwrite("bentest.mat", Dict("gt" => Matrix(transpose(X)), "D1" => A, "ob" => Matrix(transpose(Y))))

# dinds = CartesianIndices(A)

# γ = exp(0.25)
γ = 1.0
η = 0.01

# A_init = MLEM_A(a_dim, 25, Y, H, m0, P, Q, R)
A_init = rand(a_dim, a_dim)
A_init = (A_init + A_init')/2

# f_list = [x -> _spec(x, 0.99), x -> η .* _laplace(x)]
# f_list = [x -> η .* _laplace(x)]
#
# # p_list = [(x,y) -> proj_spec(x, 0.99), (x,y) -> prox_laplace(x, η * y)]
# p_list = [(x,y) -> prox_laplace(x, η * y)]

# a_gem1 = graphEM_MS(
#     a_dim,
#     30,
#     Y,
#     H,
#     m0,
#     P,
#     Q,
#     R,
#     0.9 / 2;
#     f_list = f_list,
#     p_list = p_list,
# ); display(a_gem1)

a_gem1 = GraphEM_stable(Y, H, Q, R, m0, P, r = 20, λ = 0.99/3)

okalm_n = _kalman(Y, A, H, Q, R, m0, P, likelihood = true)
okalm_o = _perform_kalman(Y, A, H, m0, P, Q, R, lle = true)


# a_gem = graphEM(a_dim, 50, Y, H, m0, P, Q, R, γ = γ, θ = 1.0, init = vec(A_init)); display(a_gem)
# a_gem_clstr = graphem_clustering(4, a_dim, 50, Y, H, m0, P, Q, R, γ = γ, θ = 1.0, directed = true, max_iters = 50, init = vec(A_init), rand_reinit = true)
a_gem_clstr = Stable_GraphEM_clustering(4, Y, H, Q, R, m0, P, rand_reinit = true)

true_filtered = _perform_kalman(Y, A, H, m0, P, Q, R)
# true_filtered = _kalman(Y, A, H, Q, R, m0, P)
reduction_filtered = _perform_kalman(Y, a_gem_clstr[1][end], H, m0, P, Q, R)
gem_filtered = _perform_kalman(Y, a_gem, H, m0, P, Q, R)

sq_error_true = cumsum(sqrt.(vec(sum((true_filtered[1] .- X).^2, dims = 1))))
sq_error_redu = cumsum(sqrt.(vec(sum((reduction_filtered[1] .- X).^2, dims = 1))))
sq_error_gem = cumsum(sqrt.(vec(sum((gem_filtered[1] .- X).^2, dims = 1))))

dtr_y = (sq_error_redu .+ sq_error_true .+ sq_error_gem) ./ 3
dtr_x = hcat(repeat([1], T), collect(1:T))
dtr_β = dtr_x \ dtr_y
dtr_fitted = dtr_x * dtr_β



f = Figure(resolution = (1600, 800))
ax_true = Axis(f[1,1], title = "Cumulative RtSq. Error")
scatterlines!(1:T, sq_error_true, label = "True Filter")
scatterlines!(1:T, sq_error_redu, label = "Reduced Filter")
scatterlines!(1:T, sq_error_gem, label = "GraphEM Filter")
lines!(1:T, dtr_fitted, label = "Trend", color = "red")
xlims!(1, T)
axislegend(position=:rb)
f

f1 = Figure(resolution = (1600, 800))
ax_true = Axis(f1[1,1], title = "Detrended Cumulative RtSq. Error")
lines!(1:T, sq_error_true .- dtr_fitted, label = "True Filter")
lines!(1:T, sq_error_redu .- dtr_fitted, label = "Reduced Filter")
lines!(1:T, sq_error_gem .- dtr_fitted, label = "GraphEM Filter")
xlims!(1, T)
axislegend(position=:lb)
f1

f2 = Figure(resolution = (1600, max(1600, 200*a_dim)))
for dimension in 1:a_dim
    ax_temp = Axis(f2[dimension, 1])
    scatter!(ax_temp, 1:T, X[dimension,:], label = "True State", color = "mediumpurple1")
    lines!(ax_temp, 1:T, true_filtered[1][dimension,:], label = "TF State")
    lines!(ax_temp, 1:T, reduction_filtered[1][dimension,:], label = "RF State")
    lines!(ax_temp, 1:T, gem_filtered[1][dimension,:], label = "GE State")
    xlims!(0, T+1)
    if dimension == a_dim
        Legend(f2[1:a_dim, 2], ax_temp)
    end
end
rowgap!(f2.layout, 1)

sum((a_gem_clstr[1][end] .- A).^2)
sum((a_gem .- A).^2)
# sum((A_init .- A).^2)

f1
f2
