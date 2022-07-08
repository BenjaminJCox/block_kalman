using LinearAlgebra
using OffsetArrays #no need to make things complicated!
using Kronecker
using Distributions

function _kalman(y, A, H, Q, R, μ₀, Σ₀; drop_priors = true, likelihood = false)
    K = size(y, 2)
    _state_dim = size(A, 1)
    μ = OffsetArray(Matrix{Float64}(undef, _state_dim, K + 1), OffsetArrays.Origin(1, 0))
    Σ = OffsetArray(Array{Float64}(undef, _state_dim, _state_dim, K + 1), OffsetArrays.Origin(1, 1, 0))

    μ[:, 0] = μ₀
    Σ[:, :, 0] = Σ₀

    if likelihood
        ll_est = 0.0
    end

    for k = 1:K
        # prediction
        μₖ = A * μ[:, k-1]
        Σₖ = A * Σ[:, :, k-1] * transpose(A) + Q

        # updating
        νₖ = H * μₖ #\nu
        vₖ = y[:, k] - νₖ #v
        Sₖ = H * Σₖ * transpose(H) + R
        Kₖ = Σₖ * transpose(H) * inv(Sₖ)

        μ[:, k] = μₖ + Kₖ * vₖ
        Σ[:, :, k] = Σₖ - Kₖ * Sₖ * transpose(Kₖ)

        if likelihood
            ll_est += logpdf(MvNormal(H * μₖ, 0.5 .* (Sₖ .+ transpose(Sₖ))), y[:, k])
        end
    end

    if drop_priors
        # for general use
        if !likelihood
            return (Matrix(μ[:, 1:end]), Array(Σ[:, :, 1:end]))
        else
            return (Matrix(μ[:, 1:end]), Array(Σ[:, :, 1:end]), ll_est)
        end
    else
        # use with RTS smoother
        return (μ, Σ)
    end
end

function _RTSS(y, A, H, Q, R, μ₀, Σ₀)
    μ, Σ = _kalman(y, A, H, Q, R, μ₀, Σ₀, drop_priors = false)

    K = size(y, 2)
    _state_dim = size(A, 1)

    μˢ = OffsetArray(Matrix{Float64}(undef, _state_dim, K + 1), OffsetArrays.Origin(1, 0))
    Σˢ = OffsetArray(Array{Float64}(undef, _state_dim, _state_dim, K + 1), OffsetArrays.Origin(1, 1, 0))
    G = OffsetArray(Array{Float64}(undef, _state_dim, _state_dim, K + 1), OffsetArrays.Origin(1, 1, 0))

    μˢ[:, K] = μ[:, K]
    Σˢ[:, :, K] = Σ[:, :, K]
    G[:, :, K] = Σ[:, :, T] * A' * inv(A * Σ[:, :, K] * A' + Q)

    for k = K-1:-1:0
        μ̄ = A * μ[:, k]
        Σ̄ = A * Σ[:, :, k] * A' + Q

        G[:, :, k] = Σ[:, :, k] * A' * inv(Σ̄)

        μˢ[:, k] = μ[:, k] + G[:, :, k] * (μˢ[:, k+1] - μ̄)
        Σˢ[:, :, k] = Σ[:, :, k] + G[:, :, k] * (Σ[:, :, k+1] - Σ̄) * G[:, :, k]'
    end
    return (μˢ, Σˢ, G, _ΨΔΦ((μˢ, Σˢ, G)))
end

function _ΨΔΦ(RTSS)
    μ, Σ, G = RTSS

    K = size(μ, 2) - 1
    n = size(μ, 1)

    Φ = zeros(n, n)
    Δ = zeros(n, n)
    Ψ = zeros(n, n)

    for k = 1:K
        Ψ .+= parent(Σ[:, :, k] .+ μ[:, k] * μ[:, k]')
        Δ .+= parent(Σ[:, :, k] * G[:, :, k-1] .+ μ[:, k] * μ[:, k-1]')
        Φ .+= parent(Σ[:, :, k-1] .+ μ[:, k-1] * μ[:, k-1]')
    end
    Φ ./= K
    Δ ./= K
    Ψ ./= K
    return (Ψ, Δ, Φ)
end

function _f1(A, Q, RTSS)
    K = size(RTSS[1], 2)
    Ψ, Δ, Φ = RTSS[4]
    return (K / 2) * tr(inv(Q) * (Ψ - Δ * A' - A * Δ' + A * Φ * A'))
    # return (1 / 2) * tr(inv(Q) * (Ψ - Δ * A' - A * Δ' + A * Φ * A'))
end

allequal(x) = all(y->y==x[1],x)

function _prox_f1(A, ϑ, Q, RTSS)
    Ψ, Δ, Φ = RTSS[4]
    K = size(RTSS[1], 2) - 1
    # return sylvester(ϑ * inv(Q), inv(Φ), -(A * inv(Φ) + ϑ * inv(Q) * Δ * inv(Φ)))

    #
    # Q_inv = inv(Q)
    # Φ_inv = inv(Φ)
    #
    C = Δ

    if isdiag(Q) && allequal(diag(Q))
        temp = ϑ * K / (Q[1]^2)
        # @info("diag shortcut")
        return (temp * C + A) * pinv(Φ * temp + Matrix(I(size(Q, 1))))
    end

    # return sylvester(K * inv(Q), ϑ * inv(Φ), -(K * inv(Q) * Δ * inv(Φ)))

    # id = 1.0 .* Matrix(I(size(Q, 1)))
    #
    # _t1 = id ⊗ (K * Q_inv) .+ (ϑ * Φ_inv) ⊗ id
    # _t2 = vec(K * Q_inv * C * Φ_inv)
    # rv = inv(_t1) * _t2
    # # @info("doody")
    # # rvl = isqrt(length(rv))
    # return reshape(rv, size(A))
end

function _laplace(A, r)
    return r .* sum(abs.(A))
end

function _prox_laplace(A, r, γ)
    α = r * γ
    return sign.(A) .* max.(abs.(A) .- α, 0.0)
end

function _prox_stable(A, η)
    U, s, V = svd(A)
    # S = diagm(min.(s, η))
    # return U * S * V'
    return U * diagm(sign.(s) .* min.(abs.(s), η)) * V'
end

function _create_adjacency_AR1(N, ρ)
    A = zeros(N, N)
    for j = 1:N, i = 1:N
        A[i, j] = ρ^abs((i - 1) - (j - 1))
    end
    return A
end

function _MS_l1_stab(A, Q, _RTSS; γ = 0.663, η = 0.99, r = 0.01, MS_iterations = 1000, ξ = 1e-4, dense_indices = eachindex(A))
    # X = V1 = V2 = P1 = P1p = A
    X = copy(A)

    V1 = copy(A)
    V2 = copy(A)

    P1 = copy(A)
    P21 = copy(A)
    P22 = copy(A)
    P1p = copy(A)

    Y1 = copy(A)
    Y21 = copy(A)
    Y22 = copy(A)

    Q1 = copy(A)
    Q21 = copy(A)
    Q22 = copy(A)
    # Ψ, Δ, Φ = _RTSS
    @inline majorant(M) = _f1(M, Q, _RTSS) .+ _laplace(M, r)
    @inline objective(A1, A2) = abs(majorant(A1) .- majorant(A2))
    for iter in 1:MS_iterations
        Y1[dense_indices] = X[dense_indices] .- γ .* (V1[dense_indices] .+ V2[dense_indices])
        Y21[dense_indices] = V1[dense_indices] .+ γ .* X[dense_indices]
        Y22[dense_indices] = V2[dense_indices] .+ γ .* X[dense_indices]

        # P1p = _prox_laplace(Y1, γ, r)
        # if iter > 1
        #     if objective(P1, _prox_laplace(Y1, γ, r)) <= ξ
        #         @info("MS converged in $(iter) iterations")
        #         return _prox_laplace(Y1, γ, r)
        #     end
        # end
        #
        # P1 = copy(P1p)
        P1[dense_indices] = _prox_laplace(Y1, γ, r)[dense_indices]

        P21[dense_indices] = Y21[dense_indices] .- γ .* _prox_f1(Y21 ./ γ, 1 / γ, Q, _RTSS)[dense_indices]
        P22[dense_indices] = Y22[dense_indices] .- γ .* _prox_stable(Y22 ./ γ, η)[dense_indices]

        Q1[dense_indices] = P1[dense_indices] .- γ .* (P21[dense_indices] .+ P22[dense_indices])
        Q21[dense_indices] = P21[dense_indices] .+ γ .* P1[dense_indices]
        Q22[dense_indices] = P22[dense_indices] .+ γ .* P1[dense_indices]

        X[dense_indices] = X[dense_indices] .- Y1[dense_indices] .+ Q1[dense_indices]
        V1[dense_indices] = V1[dense_indices] .- Y21[dense_indices] .+ Q21[dense_indices]
        V2[dense_indices] = V2[dense_indices] .- Y22[dense_indices] .+ Q22[dense_indices]
    end
    # @warn("MS did not converge before $(MS_iterations) iterations")
    return P1
end

function _MS_l1(A, Q, _RTSS; γ = 0.505, r = 0.01, MS_iterations = 1000, ξ = 1e-4)
    # X = V = P1 = P1p = A
    X = copy(A)
    V = copy(A)
    P1 = copy(A)
    P1p = copy(A)
    # Ψ, Δ, Φ = _RTSS
    @inline majorant(M) = _f1(M, Q, _RTSS) .+ _laplace(M, r)
    @inline objective(A1, A2) = abs(majorant(A1) .- majorant(A2))
    for iter in 1:MS_iterations
        Y1 = X .- γ .* V
        Y2 = V .+ γ .* X

        # P1p = _prox_laplace(Y1, γ, r)
        # if iter > 5
        #     if objective(P1, P1p) <= ξ
        #         @info("MS converged in $(iter) iterations")
        #         return P1p
        #     end
        # end

        # P1 .= P1p
        P1 = _prox_laplace(Y1, γ, r)
        P2 = Y2 .- γ .* _prox_f1(Y2 ./ γ, 1 / γ, Q, _RTSS)

        Q1 = P1 .- γ .* P2
        Q2 = P2 .+ γ .* P1

        X = X .- Y1 .+ Q1
        V = V .- Y2 .+ Q2
    end
    # @warn("MS did not converge before $(MS_iterations) iterations")
    return P1
end

function _DR_l1(A, Q, _RTSS; r = 0.01, DR_iterations = 1000, ξ = 1e-4)
    V = copy(A)
    Y = copy(A)
    X = copy(A)
    Xp = copy(A)
    # Ψ, Δ, Φ = _RTSS
    @inline majorant(M) = _f1(M, Q, _RTSS) .+ _laplace(M, r)
    @inline objective(A1, A2) = abs(majorant(A1) .- majorant(A2))
    for iter in 1:DR_iterations
        Xp = _prox_laplace(Y, 1, r)

        if iter > 1
            if objective(X, Xp) <= ξ
                @info("DR converged in $(iter) iterations")
                return Xp
            end
        end

        X .= Xp

        V = _prox_f1(2 .* X .- Y, 1, Q, _RTSS)

        Y = Y .+ V .- X
    end
    @warn("DR did not converge before $(DR_iterations) iterations")
    return X
end

function GraphEM_stable(
    y,
    H,
    Q,
    R,
    μ₀,
    Σ₀;
    λ = 0.33,
    γ = (1-λ)*0.99,
    η = 0.99,
    r = 2.0,
    A₀ = _prox_stable(_create_adjacency_AR1(size(Q, 1), 0.1) .+ 0 .* randn(size(Q)), η),
    E_iterations = 50,
    M_iterations = 1000,
    ϵ = 1e-3,
    ξ = 1e-6,
    dense_indices = eachindex(A₀)
)
    # A = Ap = A₀
    A = copy(A₀)
    Ap = copy(A₀)
    A[Not(dense_indices)] .= 0.0
    Ap[Not(dense_indices)] .= 0.0
    # @info(A₀)
    for s in 1:E_iterations
        RTSS_output = _RTSS(y, A, H, Q, R, μ₀, Σ₀)
        Ap = _MS_l1_stab(A, Q, RTSS_output; γ = γ, η = η, r = r, MS_iterations = M_iterations, ξ = ξ, dense_indices = dense_indices)
        # Ap = _MS_l1(A, Q, RTSS_output; γ = γ, r = r, MS_iterations = M_iterations, ξ = ξ)
        # Ap = _DR_l1(A, Q, RTSS_output; r = r, DR_iterations = M_iterations, ξ = ξ)
        if s > 1
            if norm(A .- Ap, 2) < ϵ * norm(A)
                @info("GraphEM converged in $(s) EM iterations")
                return Ap
            end
        end
        A .= Ap
        # @info(A)
    end
    @warn("GraphEM did not converge after $(E_iterations) EM iterations")
    return A
end
