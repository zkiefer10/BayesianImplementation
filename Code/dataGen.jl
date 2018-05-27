using Distributions
using Plots

# Parameters of data-generating process
beta_0 = 10
beta_1 = 5
N = 500

# Declare x-generating distribution
uni = Uniform(0,50)

# Generate x
x = rand(uni, N)

# Calculate parameters for (r,p) parameterization of negative binomial distribution
# r is fixed; mu is a function of x and varies for each observation
true_mu = beta_0 + beta_1 * x
true_r = 10
p = 1 .- true_mu ./ (true_r .+ true_mu)

# Declare space for y data
y = zeros(N)

# Generate y observations from negative binomial distribution
for i in 1:N
    negBinom = NegativeBinomial(true_r, p[i])
    y[i] = rand(negBinom)
end

# Plot data for visual confirmation
scatter(x, y)

# Save data for later use
writecsv("./Data/data.csv", [x y])
