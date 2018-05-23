using Distributions
using Plots

# def nb_acceptance_ratio(theta, theta_p, y, N):
#     """ theta = (mu, r), y is data, N = len(x) """
#     mu, r = theta
#     mu_p, r_p = theta_p
#
#     term1 =  r_p * np.log(r_p / (r_p + mu_p))
#     term2 = -r * np.log(r / (r + mu))
#
#     term3 = y * np.log(mu_p / mu * (mu + r)/(mu_p + r_p))
#
#     term4 = gammaln(r_p + y)
#     term5 = - gammaln(r + y)
#
#     term6 = N * (gammaln(r) - gammaln(r_p))
#
#     return (term1 + term2 + term3 + term4 + term5).sum() + term6

function nbAcceptanceRatio(mu, r, mu_p, r_p, y, N)
    term1 = r_p * log(r_p/(r_p + mu_p))
    term2 = -4 * log(r/(r + mu))
    term3 = y * log(mu_p / mu * (mu + r)/(mu_p + r_p))
    term4 = log(gamma(r_p + y))
    term5 = -log(gamma(r + y))
    term6 = N * (log(gamma(r)) - log(gamma(r_p)))

    return
end

beta0 = 10
beta1 = 5
N = 150

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
