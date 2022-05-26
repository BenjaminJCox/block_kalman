using DrWatson
using Distributions, Random, LinearAlgebra, BlockDiagonals
using InvertedIndices

include(srcdir("linear_filter_smoother.jl"))
include(srcdir("lgssm_clustering.jl"))


Random.seed!(0xabcdefabcf)

A_blocks = [rand(2, 2) .+ 2.0 for _ = 1:3]
a_dim = 6
A = BlockDiagonal(A_blocks)
A = Matrix(A)
A ./= 1.1 .* eigmax(A)

Q = Matrix(1.0 .* I(a_dim))
R = Matrix(1.0 .* I(a_dim))
# Q = Matrix(1e-2 .* I(a_dim))
# R = Matrix(1e-2 .* I(a_dim))
H = Matrix(1.0 .* I(a_dim))
P = Matrix(1e-8 .* I(a_dim))

m0 = ones(a_dim)

T = 150

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

dinds = CartesianIndices(A)

# a_gem = graphEM(a_dim, 30, Y, H, m0, P, Q, R, γ = 0.9)
a_gem_clstr = graphem_clustering(3, a_dim, 30, Y, H, m0, P, Q, R, γ = 0.9)
