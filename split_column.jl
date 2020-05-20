# split Xa_all into different parts for parallel

using JLD


nbatch =2
data_path = "/home/tianjing/BayesXII_data/demo/"

Xa_path = data_path*"Xa.jld"
Xa_data = load(Xa_path);
Xa      = Xa_data["Xa"];

nCol      = size(Xa,2)
bsize     = Int(floor(nCol/nbatch))
ncol_rank = Int32.(zeros(nbatch))

for i in 1:nbatch
    istart = (i-1)*bsize+1
    if i == nbatch
        iend = nCol
    else
        iend = i*bsize
    end
    Xa_part      = Xa[:,istart:iend]
    irank        = i-1  #rank starts from 0
    ncol_rank[i] = size(Xa_part,2)
    save(data_path*"$nbatch/Xa$irank.jld","Xa",Xa_part)
end

save(data_path*"$nbatch/ncol_rank.jld","ncol_rank",ncol_rank)