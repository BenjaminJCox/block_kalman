using LinearAlgebra
using Distributions
using DrWatson
using Kronecker

function _perform_kalman(observations, A, H, m0, P0, Q, R; lle = false)
    m = copy(m0)
    P = copy(P0)
    v = observations[:, 1] .- H * m
    S = H * P * H' + R
    K = P * H' / S
    T = size(observations, 2)
    _xd = length(m0)
    filtered_state = zeros(length(m0), T)
    filtered_cov = zeros(length(m0), length(m0), T)
    l_like_est = 0.0
    # offness = 0.0
    for t = 1:T
        m .= A * m
        P .= A * P * transpose(A) .+ Q
        v .= observations[:, t] .- H * m
        S .= H * P * transpose(H) .+ R
        # offness += norm(S - Matrix(Hermitian(S)), 1)
        S .= 0.5 .* (S .+ transpose(S))
        K .= (P * transpose(H)) * inv(S)
        if lle
            l_like_est += logpdf(MvNormal(H * m, S), observations[:, t])
        end
        # unstable, need to implement in sqrt form
        m .= m .+ K * v
        P .= (I(_xd) .- K * H) * P * (I(_xd) .- K * H)' .+ K * R * K'
        filtered_state[:, t] .= m
        filtered_cov[:, :, t] .= P
    end
    return (filtered_state, filtered_cov, l_like_est, (m0, P0))
end

function _perform_rts(kalman_out, A, H, Q, R)
    kal_means = kalman_out[1]
    kal_covs = kalman_out[2]
    pri = kalman_out[4]
    T = size(kal_means, 2)
    rts_means = zeros(size(kal_means))
    rts_covs = zeros(size(kal_covs))
    rts_means[:, T] = kal_means[:, T]
    rts_covs[:, :, T] = kal_covs[:, :, T]
    # just preallocation, values not important
    m_bar = A * kal_means[:, T]
    P_bar = A * kal_covs[:, :, T] * A' + Q
    G = kal_covs[:, :, T] * A' / P_bar
    G_ks = zeros(size(G)..., T)
    G_ks[:, :, T] .= G
    m = copy(m_bar)
    P = copy(P_bar)
    zs_G = G
    zs_m = m
    zs_P = P
    for k = (T-1):-1:0
        if k != 0
            m_bar .= A * kal_means[:, k]
            P_bar .= A * kal_covs[:, :, k] * A' .+ Q
            G .= kal_covs[:, :, k] * A' / P_bar
            G_ks[:, :, k] .= G
            m .= kal_means[:, k] .+ G * (rts_means[:, k+1] .- m_bar)
            P .= kal_covs[:, :, k] .+ G * (rts_covs[:, :, k+1] .- P_bar) * G'
            rts_means[:, k] .= m
            rts_covs[:, :, k] .= P
        else
            m_bar .= A * pri[1]
            P_bar .= A * pri[2] * A' .+ Q
            G .= pri[2] * A' / P_bar
            zs_G = G
            m .= pri[1] .+ G * (rts_means[:, k+1] .- m_bar)
            P .= pri[2] .+ G * (rts_covs[:, :, k+1] .- P_bar) * G'
            zs_m .= m
            zs_P .= P
        end
    end
    return (rts_means, rts_covs, G_ks, (zs_G, zs_m, zs_P))
end

function _Q_func(observations, A, H, m0, P0, Q, R)
    kal = _perform_kalman(observations, A, H, m0, P0, Q, R, lle = false)
    rts = _perform_rts(kal, A, H, Q, R)
    rts_means = rts[1]
    rts_covs = rts[2]
    rts_G = rts[3]
    rts_z = rts[4]

    Σ = zeros(size(P0))
    Φ = zeros(size(P0))
    B = zeros(size(observations[:, 1] * rts_means[:, 1]'))
    C = zeros(size(m0 * m0'))
    D = zeros(size(observations[:, 1] * observations[:, 1]'))

    K = size(observations, 2)

    B += observations[:, 1] * rts_means[:, 1]'
    Σ += rts_covs[:, :, 1] .+ (rts_means[:, 1] * rts_means[:, 1]')
    Φ += rts_z[3] + (rts_z[2] * rts_z[2]')
    C += (rts_covs[:, :, 1] * rts_z[1]') .+ (rts_means[:, 1] * rts_z[2]')
    D += observations[:, 1] * observations[:, 1]'

    for k = 2:K
        B += observations[:, k] * rts_means[:, k]'
        Σ += rts_covs[:, :, k] .+ (rts_means[:, k] * rts_means[:, k]')
        Φ += rts_covs[:, :, k-1] .+ (rts_means[:, k-1] * rts_means[:, k-1]')
        C += (rts_covs[:, :, k] * rts_G[:, :, k-1]') + (rts_means[:, k] * rts_means[:, k-1]')
        D += observations[:, k] * observations[:, k]'
    end
    B ./= K
    Σ ./= K
    Φ ./= K
    C ./= K
    D ./= K

    val_dict = @dict Σ Φ C B D
    return val_dict
end

function f1_fvd(A, Q, val_dict)
    Σ = val_dict[:Σ]
    C = val_dict[:C]
    Φ = val_dict[:Φ]
    return 0.5 .* tr(inv(Q) * (Σ - C * A' - A * C' + A * Φ * A'))
end


function _Q_func(observations, A′, H, m0, P0, Q, R, _lp)
    kal = _perform_kalman(observations, A′, H, m0, P0, Q, R)
    rts = _perform_rts(kal, A′, H, Q, R)
    rts_means = rts[1]
    rts_covs = rts[2]
    rts_G = rts[3]
    rts_z = rts[4]

    Σ = zeros(size(P0))
    Φ = zeros(size(P0))
    B = zeros(size(observations[:, 1] * rts_means[:, 1]'))
    C = zeros(size(m0 * m0'))
    D = zeros(size(observations[:, 1] * observations[:, 1]'))

    K = size(observations, 2)

    B .+= observations[:, 1] * rts_means[:, 1]'
    Σ .+= rts_covs[:, :, 1] .+ (rts_means[:, 1] * rts_means[:, 1]')
    Φ .+= rts_z[3] .+ (rts_z[2] * rts_z[2]')
    C .+= (rts_covs[:, :, 1] * rts_z[1]') .+ (rts_means[:, 1] * rts_z[2]')
    D .+= observations[:, 1] * observations[:, 1]'

    for k = 2:K
        B .+= observations[:, k] * rts_means[:, k]'
        Σ .+= rts_covs[:, :, k] .+ (rts_means[:, k] * rts_means[:, k]')
        Φ .+= rts_covs[:, :, k-1] .+ (rts_means[:, k-1] * rts_means[:, k-1]')
        C .+= (rts_covs[:, :, k] * rts_G[:, :, k-1]') .+ (rts_means[:, k] * rts_means[:, k-1]')
        D .+= observations[:, k] * observations[:, k]'
    end
    B ./= K
    Σ ./= K
    Φ ./= K
    C ./= K
    D ./= K

    _f1(A) = (K ./ 2.0) .* tr(inv(Q) * (Σ .- C * A' .- A * C' .+ A * Φ * A'))
    _f2(A) = _lp(A)
    Qf(A) = _f1(A) .+ _f2(A)
    val_dict = @dict Σ Φ C B D
    return (Qf, _f1, _f2, val_dict)
end

function _oneshot_A(val_dict)
    C = val_dict[:C]
    Φ = val_dict[:Φ]
    A_map = C / Φ
    return A_map
end

function MLEM_A(dimA, steps, Y, H, m0, P, Q, R)
    A_est = rand(dimA, dimA)
    for step = 1:steps
        VD = _Q_func(Y, A_est, H, m0, P, Q, R)
        A_est = _oneshot_A(VD)
    end
    return A_est
end

function _proxf1(A, θ, K, Q, val_dict)
    C = val_dict[:C]
    Φ = val_dict[:Φ]
    Q_inv = inv(Q)
    Φ_inv = inv(Φ)

    id = 1.0 .* Matrix(I(size(Q, 1)))

    _t1 = id ⊗ (K * Q_inv) .+ (θ * Φ_inv) ⊗ id
    _t2 = vec(K * Q_inv * C * Φ_inv)
    rv = inv(_t1) * _t2
    # rvl = isqrt(length(rv))
    return reshape(rv, size(A))
end

function _proxf2(A, θ)
    maximator = max.(abs.(A) .- θ, 0.0)
    return sign.(A) .* maximator
end

function prox_f1(A, Q, ϑ, val_dict)
    # in new notation C is now Δ, using old for consistency
    C = val_dict[:C]
    Φ = val_dict[:Φ]
    Q_inv = inv(Q)
    Φ_inv = inv(Φ)

    # id = 1.0 .* Matrix(I(size(Q, 1)))
    #
    # _t1 = id ⊗ (ϑ * Q_inv) .+ (Φ_inv) ⊗ id
    # _t2 = vec(ϑ * Q_inv * C * Φ_inv)
    #
    # rv = inv(_t1) * _t2
    # return reshape(rv, size(A))
    return sylvester(ϑ .* Q_inv, Φ_inv, A * Φ_inv .+ ϑ .* Q_inv * C * Φ_inv)
end

function prox_laplace(A, ϑ)
    return sign.(A) .* max.(abs.(A) .- ϑ, 0.0)
end

function _laplace(A)
    return sum(abs.(A))
end

function prox_gaussian(A, ϑ)
    return A ./ (1.0 + ϑ)
end

function _gaussian(A)
    return 0.5 .* sum(A .^ 2)
end

function prox_lapgauss(A, ϑ)
    return sign.(A ./ (1.0 + ϑ)) .* max.(abs.(A ./ (1.0 + ϑ)) .- ϑ / (1.0 + ϑ), 0.0)
end

function _lapgauss(A)
    return _laplace(A) + _gaussian(A)
end

function proj_spec(A, δ)
    U, s, V = svd(A)
    return U * diagm(sign.(s) .* min.(abs.(s), δ)) * V
end

function _spec(A, δ)
    # return 1.0 * (eigmax(A) <= δ)
    return 0.0
end

function proj_range(A, α, β)
    return min.(max.(A, α), β)
end

function _range(A)
    return 0.0
end

function _MS_opt(;
    λ::Number,
    γ::Number,
    f_list::Vector,
    p_list::Vector,
    A₀::Matrix,
    ϵ::AbstractFloat,
    dense_indices,
    max_iters::Int = 100,
)
    M = length(f_list)
    V = [A₀ for m = 1:M]
    W = [zero(A₀) for m = 1:M]
    A = [zero(A₀) for m = 1:M]
    Z = [zero(A₀) for m = 1:M]

    @inline Q(A) = sum([f(A) for f in f_list])

    iters = 0

    difference = 2ϵ
    old_Q::Float64 = Inf
    new_Q::Float64 = 0.0
    while (difference >= ϵ) && (iters < max_iters)

        for w_ind in 1:(M-1)
            W[w_ind][dense_indices] .= V[w_ind][dense_indices] .+ λ .* V[M][dense_indices]
        end

        W[M][dense_indices] .= V[M][dense_indices] .- γ .* sum(V[1:(M-1)])[dense_indices]

        for a_ind in 1:(M-1)
            A[a_ind][dense_indices] .= W[a_ind][dense_indices] .- λ .* p_list[a_ind](W[a_ind], inv(γ))[dense_indices]
        end
        A[M][dense_indices] .= p_list[M](W[M], γ)[dense_indices]

        # @info Q(A[M])
        new_Q = Q(A[M])
        difference = abs(new_Q .- old_Q)
        old_Q = copy(new_Q)
        iters += 1

        for z_ind in 1:(M-1)
            Z[z_ind][dense_indices] .= A[z_ind][dense_indices] .+ λ .* A[M][dense_indices]
        end
        Z[M][dense_indices] .= A[M][dense_indices] .- γ .* sum(A[1:(M-1)])[dense_indices]

        for v_ind in 1:M
            V[v_ind][dense_indices] .= V[v_ind][dense_indices] .- W[v_ind][dense_indices] .+ Z[v_ind][dense_indices]
        end
    end
    return A[M]
end


function _DR_opt(f1, f2, proxf1, proxf2, θ, K, Q, val_dict, Z0, ϵ, γ; max_iters = 100, dense_indices = eachindex(Z0))
    difference = 2 * ϵ
    Z = copy(Z0)
    A = zero(Z)
    A[dense_indices] .= proxf2(Z, θ)[dense_indices]
    A_old = copy(A)
    V = zero(A)
    V[dense_indices] .= proxf1(2A .- Z, θ, K, Q, val_dict)[dense_indices]
    Z[dense_indices] .= (Z+θ.*(V.-A))[dense_indices]
    # println(Z + θ .* (V - A))
    iters = 0
    while (difference >= ϵ) && (iters < max_iters)
        A[dense_indices] .= proxf2(Z, γ)[dense_indices]
        V[dense_indices] .= proxf1(2.0 .* A .- Z, θ, K, Q, val_dict)[dense_indices]
        # @info(V, isotropic_proxf1(2.0 .* A - Z, θ, K, Q, val_dict))
        Z[dense_indices] .= (Z.+θ.*(V.-A))[dense_indices]
        difference = abs.(f1(A) + f2(A) - f1(A_old) - f2(A_old))
        A_old .= A
        iters += 1
    end
    return A
end

function graphEM_MS(
    dimA,
    steps,
    Y,
    H,
    m0,
    P,
    Q,
    R,
    λ;
    dense_indices = eachindex(zeros(dimA, dimA)),
    init = rand(length(dense_indices)),
    max_iters = 100,
    ϵ = 1e-6,
    f_list::Vector,
    p_list::Vector,
)
    A_gem = zeros(dimA, dimA)
    A_gem[dense_indices] .= init


    for s = 1:steps
        _vd = _Q_func(Y, A_gem, H, m0, P, Q, R)
        n_fl = vcat(x -> f1_fvd(x, Q, _vd), f_list)
        n_pl = vcat((x,y) -> prox_f1(x, Q, y, _vd), p_list)
        A_gem = _MS_opt(
            λ = λ,
            γ = (1 - λ) / (length(f_list)),
            f_list = n_fl,
            p_list = n_pl,
            A₀ = A_gem,
            ϵ = ϵ,
            dense_indices = dense_indices,
            max_iters = max_iters,
        )
    end
    return A_gem
end


function graphEM(
    dimA,
    steps,
    Y,
    H,
    m0,
    P,
    Q,
    R;
    γ = 0.1,
    θ = 1.0,
    dense_indices = eachindex(zeros(dimA, dimA)),
    init = rand(length(dense_indices)),
)
    A_gem = zeros(dimA, dimA)
    A_gem[dense_indices] .= init
    T = size(Y, 2)
    @inline l1_penalty(A) = γ * norm(A, 1)
    for s = 1:steps
        Qf = _Q_func(Y, A_gem, H, m0, P, Q, R, l1_penalty)
        A_gem = _DR_opt(Qf[2], Qf[3], _proxf1, _proxf2, θ, T, Q, Qf[4], A_gem, 1e-3, γ, dense_indices = dense_indices)
    end
    return A_gem
end
