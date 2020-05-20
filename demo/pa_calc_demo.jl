using Statistics
using JLD

data_root = "/home/tianjing/BayesXII_data/demo/"
res_root = "/home/tianjing/BayesXII_res/demo/"


#X_test
X_test_path = data_root*"X_test.jld"
X_test_data = load(X_test_path);
X_test = X_test_data["X_test"];

#y_test
y_test_path = data_root*"y_test.jld"
y_test_data = load(y_test_path);
y_test = y_test_data["y_test"];

#res
res_data=load(res_root*"BayesXII.niter1000.seed123.jld")
mat_mean_alpha = res_data["mat_mean_alpha"]

# calculate prediction accuracy
pa = []
for i in eachcol(mat_mean_alpha)
    push!(pa,cor(X_test*i,y_test))
end

save(res_root*"pa.BayesXII.niter1000.seed123.jld","pa",pa)