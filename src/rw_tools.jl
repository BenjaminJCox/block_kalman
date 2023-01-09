using LinearAlgebra
using Graphs, SimpleWeightedGraphs

function rw_transition_matrix(A)
    Ap = abs.(A)
    Ap = Ap ./ sum.(eachrow(Ap))
    return Ap
end

function stationary_approx(A)
    Ap = rw_transition_matrix(A) ^ 50
    return Ap
end

function frac_stat(A)
    rws = stationary_approx(A)
    Ap = deepcopy(A)
    for idx in eachindex(A)
        if Ap[idx] != 0
            Ap[idx] = rws[idx]
        end
    end
    return Ap
end
