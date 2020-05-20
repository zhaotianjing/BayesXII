using JLD
using Statistics
using LinearAlgebra
using Arpack
using StatsBase
using Random
Random.seed!(1234)


save_path="/home/tianjing/BayesXII_data/demo/"
nTrain = 200
nTest  = 100
n      = nTrain + nTest
p      = 500
myπ    = 0.95
h2     = 0.3


function center!(X)
    nrow,ncol = size(X)
    colMeans = mean(X, dims = 1)
    BLAS.axpy!(-1,ones(nrow)*colMeans,X)
    return colMeans
end


# genotype
X       = rand([0.0,1.0,2.0],n,p)

X_train = X[1:nTrain, :]
X_test  = X[nTrain+1:end, :]
save(save_path*"X_test.jld", "X_test", X_test)

# y
nQTL = trunc(Int, (p*(1-myπ)))

varg        = 1
QTL_effects = randn(nQTL)
QTL_pos     = sample(collect(1:p),nQTL)

BV = X[:,QTL_pos]*QTL_effects
BV = BV/std(BV)*sqrt(varg)

# h2 = 0.3
vare = (1-h2)/h2*varg 
save(save_path*"vare.jld","vare", vare)

#y
y       = BV + randn(n)*sqrt(vare)
y_train = y[1:nTrain]
y_test  = y[nTrain+1:end]
save(save_path*"y_test.jld","y_test", y_test)
save(save_path*"y_train.jld","y_train", y_train)

colmean = center!(X_train);
save(save_path*"X_train.jld","X_train", X_train)

Xty = X_train'y_train
save(save_path*"Xty.jld","Xty", Xty)

#vara
vara = varg/((1 - myπ) * sum( 2 * vec(colmean / 2) .* (- vec(colmean / 2) .+ 1) ) )  #vara = var(g) / 0.05*2*sum(pi*qi)
save(save_path*"vara.jld","vara", vara)

muX       = [ones(nTrain) X_train]
MM        =muX'muX;
max_eigen = eigs(copy(MM),nev=1)[1][1]
d         = max_eigen + 0.001
BB        = Matrix{Float64}(I, p+1 , p+1)*d - copy(MM);
save(save_path*"d.jld","d", d)

Xa_all = cholesky(BB).U;
Xa_all = convert(Array, Xa_all); # (p+1)-by-(p+1) matrix

Xa = Xa_all[:,2:end]
save(save_path*"Xa.jld", "Xa", Xa)

J = Xa_all[:,1]
save(save_path*"J.jld", "J", J)