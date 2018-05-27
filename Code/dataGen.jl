using Distributions
using Plots

beta_0 = 10
beta_1 = 5
N = 500

uni = Uniform(0,50)

x = rand(uni, N)

true_mu = beta_0 + beta_1 * x
true_r = 10
p = 1 .- true_mu ./ (true_r .+ true_mu)

y = zeros(N)

for i in 1:N
    negBinom = NegativeBinomial(true_r, p[i])
    y[i] = rand(negBinom)
end

scatter(x, y)

writecsv("./Data/data.csv", [x y])
