using MPI
using LinearAlgebra
using Distributions
using Random
using JLD

#please change below path
data_root        = "/home/tianjing/BayesXII_data/demo/"        # to load data
result_save_root = "/home/tianjing/BayesXII_res/demo/" # to save result

nIter   = 1000
outFreq = 10

my_seed = 123
nSNP    = 500


function BayesXII(nIter = nIter, outFreq = outFreq, my_seed = my_seed, nSNP = nSNP, data_root = data_root, result_save_root = result_save_root)
    MPI.Init()

    comm         = MPI.COMM_WORLD
    my_rank      = MPI.Comm_rank(comm)
    cluster_size = MPI.Comm_size(comm)

    nCol = nSNP      # nMarker
    nRow = nSNP + 1  # nParameter for mu

    batch_size_path = data_root*"$cluster_size/ncol_rank.jld";
    batch_size_data = load(batch_size_path);
    batch_size      = batch_size_data["ncol_rank"];

    my_batch_size = batch_size[my_rank + 1] # because my_rank starts from 0

    if my_rank == 0
        Random.seed!(my_seed) # use my_seed to gernerate chains with different starting value for marker effects (alpha)

        ## read data
        #Xty
        Xty_path = data_root*"Xty.jld"
        Xty_data = load(Xty_path);
        Xty      = Xty_data["Xty"];

        #y
        y_train_path = data_root*"y_train.jld"
        y_train_data = load(y_train_path);
        y            = y_train_data["y_train"];

        #vara
        vara_path = data_root*"vara.jld"
        vara_data = load(vara_path);
        vara      = vara_data["vara"];

        #vare
        vare_path = data_root*"vare.jld"
        vare_data = load(vare_path);
        vare      = vare_data["vare"];

        #d
        d_path = data_root*"d.jld"
        d_data = load(d_path);
        d      = d_data["d"];

        #alpha
        α         = rand(Normal(0,vara),nCol)
        meanAlpha = zeros(nCol)

        #others
        sum_y       = sum(y)
        dot_y       = dot(y,y)
        mu          = mean(y)
        nind        = length(y)

        # output
        nOutput        = Int(floor(nIter/outFreq))
        vec_mean_vare  = zeros(nOutput)
        vec_mean_vara  = zeros(nOutput)
        vec_mean_Mu    = zeros(nOutput)
        vec_nloci      = zeros(nOutput)
        mat_mean_alpha = zeros(nCol,nOutput)
        mat_alpha      = zeros(nCol,nOutput)
        vec_mean_pi    = zeros(nOutput)
        vec_vara       = zeros(nOutput)
        vec_vare       = zeros(nOutput)
        iout = 1

        meanVare    = 0.0
        meanVara    = 0.0
        meanMu      = 0.0
        meanPi      = 0.0

    else
        Random.seed!(99)

        α         = Float64[]
        Xty       = Float64[]
        meanAlpha = Float64[]
        mu        = Float64[]
        vara      = Float64[]
        vare      = Float64[]
        d         = Float64[]
        pi        = Float64[]
    end

    Random.seed!(99)  #make sure the same seed is used in all rank

    ## read data in all ranks
    #J
    J_path = data_root*"J.jld"
    J_data = load(J_path);
    J = J_data["J"];

    #partial Xa (only a part of Xa is used in each rank)
    Xa_path = data_root*"$cluster_size/Xa$my_rank.jld";
    Xa_data = load(Xa_path);
    Xa      = Xa_data["Xa"];

    #scatter from rank0 to other ranks
    α         = MPI.Scatterv(α,batch_size,0,comm) 
    Xty       = MPI.Scatterv(Xty,batch_size,0,comm) 
    meanAlpha = MPI.Scatterv(meanAlpha,batch_size,0,comm) 

    #bcast from rank0 to other ranks
    mu          = MPI.bcast(mu,0,comm)
    vare        = MPI.bcast(vare,0,comm)
    vara        = MPI.bcast(vara,0,comm)
    d           = MPI.bcast(d,0,comm)

    #others
    nuRes       = 4
    scaleRes    = vare*(nuRes-2)/nuRes
    dfEffectVar = 4
    scaleVar    = vara*(dfEffectVar-2)/dfEffectVar

    pi_fix      = 0.95
    logPi       = log(pi_fix)
    logPiComp   = log(1.0-pi_fix)


    for iter = 1:nIter
            iIter = 1.0/iter

            # sample ya
            local_xaα = Xa * α
            ya        = MPI.Allreduce(local_xaα, MPI.SUM, comm) + J*mu + randn(nRow)*sqrt(vare)  #all rank

            # sample marker effects
            rhs       = Xty + (Xa' * ya)
            lhs       = d + vare/vara
            meanm     = rhs/lhs
            varm      = vare/lhs
            α         = meanm + randn(my_batch_size)*sqrt(varm)

            # sample indicators
            v0         = d*vare
            v1         = v0+d^2*vara
            logDelta0  = -0.5*(rhs.^2/v0 .+ log(v0)) .+ logPi
            logDelta1  = -0.5*(rhs.^2/v1 .+ log(v1)) .+ logPiComp
            probDelta1 = 1.0 ./ (exp.(logDelta0-logDelta1) .+ 1.0)
            includeit  = rand(my_batch_size) .< probDelta1
            α          = α.*includeit
            meanAlpha  += (α-meanAlpha)*iIter

            local_dot_α     = dot(α,α)
            local_dot_α_rhs = dot(α,rhs)
            local_nLoci     = sum(includeit)
            dot_α           = MPI.Reduce(local_dot_α, +,0,comm)
            dot_α_rhs       = MPI.Reduce(local_dot_α_rhs, +,0, comm)
            nLoci           = MPI.Reduce(local_nLoci, +,0, comm)

            if my_rank==0
                    ## only sampled in rank0
                    # sample intercept
                    rhs_mu  = sum_y + dot(J,ya)
                    mu      = rhs_mu/d + randn()*sqrt(vare/d)
                    meanMu += (mu - meanMu)*iIter

                    # sample vare
                    yctyc     = dot_y + dot(ya,ya) + d*mu^2 + d*dot_α - 2*mu*rhs_mu-2*dot_α_rhs
                    vare      = (yctyc + nuRes*scaleRes)/rand(Chisq(nRow+nind+nuRes))
                    meanVare += (vare - meanVare)*iIter

                    #sample vara
                    vara        =  (dot_α + dfEffectVar*scaleVar)/rand(Chisq(nLoci+dfEffectVar))
                    meanVara   += (vara - meanVara)*iIter

                    #sample pi
                    aa      = nCol - nLoci + 1
                    bb      = nLoci + 1
                    pi      = rand(Beta(aa,bb))
                    meanPi += (pi - meanPi)*iIter
            end
            
            #broadcast from rank0 to other ranks
            vara,vare,pi,mu = MPI.bcast([vara,vare,pi,mu], 0, comm)

            logPi     = log(pi)
            logPiComp = log(1.0 - pi)

            # for collecting result
            meanAlpha_all = MPI.Gatherv(meanAlpha,batch_size,0,comm)
            α_all         = MPI.Gatherv(α,batch_size,0,comm)   # delete if you do not want to sample alpha

            if (iter%outFreq == 0 && my_rank==0)
                println("Iteration ",iter,
                    ", meanVara ",meanVara,
                    ", meanVare ",meanVare,
                    ", meanMu ",meanMu,
                    ", nLoci ",nLoci,
                    ", meanPi ",meanPi)

                # save result in rank0
                vec_mean_vare[iout]    = meanVare
                vec_mean_vara[iout]    = meanVara
                vec_mean_Mu[iout]      = meanMu
                vec_nloci[iout]        = nLoci
                mat_mean_alpha[:,iout] = meanAlpha_all
                vec_mean_pi[iout]      = meanPi
                mat_alpha[:,iout]      = α_all
                vec_vare[iout]         = vare
                vec_vara[iout]         = vara
                iout += 1
            end
          end

          if my_rank ==0
                  save(result_save_root*"BayesXII.niter$nIter.seed$my_seed.jld", 
                       "vec_mean_vare",vec_mean_vare,
                       "vec_mean_vara",vec_mean_vara,
                       "vec_vare",vec_vare,
                       "vec_vara",vec_vara,
                       "vec_mean_Mu",vec_mean_Mu,
                       "vec_nloci",vec_nloci,
                       "vec_mean_pi",vec_mean_pi,
                       "mat_mean_alpha",mat_mean_alpha,
                       "mat_alpha",mat_alpha)
          end

          MPI.Finalize()
end


BayesXII()
