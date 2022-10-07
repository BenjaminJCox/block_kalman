using LinearAlgebra
using Graphs, SimpleWeightedGraphs

include("graphem_prereq.jl")

# incomplete implementation of SimpleHerm from "Higher-Order Spectral Clustering of Directed Graphs"
# requires number of clusters to be known a-priori, may not be appropriate

function graph_hermitian_adjacency(A)
    k = size(A, 1)
    ham = zeros(Complex, k, k)
    for row = 1:k
        for col = 1:k
            if A[row, col] != 0
                if A[col, row] != 0
                    ham[row, col] = Complex(1, 0)
                    # ham[col, row] = Complex(1, 0)
                else
                    ham[row, col] = Complex(0, 1)
                end
            elseif A[col, row] != 0
                ham[row, col] = Complex(0, -1)
            end
        end
    end
    return ham
end

function graph_degrees(A)
    k = size(A, 1)
    out_degrees = zeros(Int, k)
    in_degrees = zeros(Int, k)
    for i in 1:k
        out_degrees[i] = sum(A[i, :])
        in_degrees[i] = sum(A[:, i])
    end
    return Dict(:in => in_degrees, :out => out_degrees, :tot => in_degrees .+ out_degrees)
end


function graph_norm_laplacian(A)
    # degree_mat = diagm(degree(G))
    degree_mat = diagm(graph_degrees(A)[:tot])
    herm_mat = graph_hermitian_adjacency(G)

    d = size(degree_mat, 1)
    lap_mat = I(d) - degree_mat^(-0.5) * herm_mat * degree_mat^(-0.5)
end

function simple_herm(A)
    L = graph_norm_laplacian(A)
    l1 = eigen(L.vectors[:, 1])

    gist_degrees = inv.(sqrt.(graph_degrees(A)[:tot]))
end
