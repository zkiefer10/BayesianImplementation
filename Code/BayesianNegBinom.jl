using Distributions
using Plots

function nbAcceptanceRatio(mu, r, mu_p, r_p, y, N)
    term1 = r_p * log(r_p./(r_p .+ mu_p))
    term2 = -r * log(((r .+ mu)./r).^-1)
    term3 = y .* log((mu_p ./ mu) .* ((mu .+ r)./(mu_p .+ r_p)))
    term4 = log(gamma(r_p .+ y))
    term5 = -log(gamma(r .+ y))
    term6 = N * (log(gamma(r)) - log(gamma(r_p)))

    # println(term1, term2, term3, term4, term5, term6)
    return sum(term1 + term2 + term3 + term4 + term5) + term6
end

function genProposal(mu, sigma)
    truncNorm = TruncatedNormal(mu, sigma, 0, Inf)
    return rand(truncNorm)
end

function calculateMu(beta, x)
    return beta[1] + beta[2] * x
end

standNorm = Normal()
standUni = Uniform(0,1)

data = readcsv("./Data/data.csv")

X = data[1:5,1]
Y = data[1:5,2]
N = length(Y)

sigma = [0.5, 0.7, 0.5] * 1
sigma_beta_0 = sigma[1]
sigma_beta_1 = sigma[2]
sigma_r = sigma[3]

n_iter = 10000
burn_in = 4000

beta_0 = 5.
beta_1 = 5.
r = 1.

trace = zeros(n_iter + 1, 3)
trace[1,:] = [beta_0, beta_1, r]
acceptances = zeros(n_iter + 1)
acceptRate = zeros(n_iter + 1)

params_b = trace[1,:]
mu_b = calculateMu(params_b, X)

boundsFlag = 0

for i in 1:n_iter
    params_g = [rand(standNorm) * sigma_beta_0 + params_b[1]; rand(standNorm) * sigma_beta_1 + params_b[2]; genProposal(params_b[3], sigma_r)]
    mu_g = calculateMu(params_g, X)

    if ((params_g[3] < 0.1) & (boundsFlag == 0))
        println("r is 0 at iteration ", i, " !")
        boundsFlag = 1
    end

    alpha = exp(nbAcceptanceRatio(mu_b, params_b[3], mu_g, params_g[3], Y, N))
    uni = rand(standUni)
    if (uni < alpha)
        trace[i+1,:] = params_g
        params_b = params_g
        mu_b = mu_g
        acceptances[i+1] = 1
    else
        trace[i+1,:] = params_b
    end
    acceptRate[i+1] = alpha
end

# plot(trace[:,3])
