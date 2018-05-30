using Distributions
using Plots

# Calculate acceptance ratio for proposed value of theta
# _p values are from proposal
# Heavily vectorized calculations
function nbAcceptanceRatio(mu, r, mu_p, r_p, y, N)
    term1 = r_p * log(r_p./(r_p .+ mu_p))
    term2 = -r * log(((r .+ mu)./r).^-1)
    term3 = y .* log((mu_p ./ mu) .* ((mu .+ r)./(mu_p .+ r_p)))
    term4 = lgamma(r_p .+ y)
    term5 = -lgamma(r .+ y)
    term6 = N * (lgamma(r) - lgamma(r_p))

    # println(term1, term2, term3, term4, term5, term6)
    return sum(term1 + term2 + term3 + term4 + term5) + term6
end

function nbPostPredict(trace, x)
    # Define size parameters
    predictPerDraw = 1
    N = length(trace[:,1])
    K = length(trace[1,:])

    # Define space for recording draws of y
    yDraws = zeros(N * predictPerDraw)

    # Draw y repeatedly
    for i in 1:N
        # First draw from the parameter distribution
        drawSpace = trace[i,:]

        # Calculate paramters of the negative binomial distribution
        mu = calculateMu(drawSpace[1:K-1], x)
        r = drawSpace[K]
        p = 1 - mu / (r + mu)

        # Draw and record set of y's
        yDraws[(i - 1) * predictPerDraw + 1:i * predictPerDraw] = rand(NegativeBinomial(r,p), predictPerDraw)
    end
    return yDraws
end

# Generate a proposal value of r from a truncated normal distribution
# r must be positive
function genProposal(mu, sigma)
    truncNorm = TruncatedNormal(mu, sigma, 0, Inf)
    return rand(truncNorm)
end

# Calculate vector of mu values from x and beta parameters
function calculateMu(beta, x)
    return beta[1] + beta[2] * x
end

function prior(x)
    # Uninformative prior:
    return 1

    #Informative prior, in which beta_0, beta_1 and r are distributed independently:
    # beta_0 = x[1]
    # beta_1 = x[2]
    # r = x[3]
    # return pdf(Normal(5, 3), beta_0) * pdf(Normal(5, 3), beta_1) * pdf(Gamma(5,3), r)
end

# Declare distributions for later use
standNorm = Normal()
standUni = Uniform(0,1)

# Read and separate data
data = readcsv("./Data/data.csv")

X = data[:,1]
Y = data[:,2]
N = length(Y)

# Declare variances for proposal distributions
sigma = [0.5, 0.7, 0.5] * 1 # Use multiplier at end to quickly calibrate acceptance rate.
sigma_beta_0 = sigma[1]
sigma_beta_1 = sigma[2]
sigma_r = sigma[3]

# Set number of burn-in and productive iterations
n_iter = 100000
burn_in = 20000

# Set initial values
beta_0 = 5.
beta_1 = 5.
r = 1.

# Declare spaces for record-keeping
trace = zeros(n_iter + 1, 3)
proposals = deepcopy(trace)
trace[1,:] = [beta_0, beta_1, r]
acceptances = zeros(n_iter + 1)
acceptRate = zeros(n_iter + 1)

# Initialize existing values before first loop
params_b = trace[1,:]
mu_b = calculateMu(params_b, X)

for i in 1:n_iter
    # Generate proposal, calculate new mu from proposal, record proposal
    params_g = [rand(standNorm) * sigma_beta_0 + params_b[1]; rand(standNorm) * sigma_beta_1 + params_b[2]; genProposal(params_b[3], sigma_r)]
    mu_g = calculateMu(params_g, X)
    proposals[i+1,:] = params_g

    # Calculate acceptance probability
    alpha = exp(nbAcceptanceRatio(mu_b, params_b[3], mu_g, params_g[3], Y, N))

    # If using an informative prior, implement that in function prior() above for use here:
    alpha = alpha * prior(params_b) / prior(params_g  )

    # Randomly accept or reject proposal, record applicable draw from distribution
    uni = rand(standUni)
    if (uni < alpha)
        trace[i+1,:] = params_g
        params_b = deepcopy(params_g)
        mu_b = deepcopy(mu_g)
        acceptances[i+1] = 1
    else
        trace[i+1,:] = params_b
    end

    # Record acceptance rates (not actual acceptance decision)
    acceptRate[i+1] = alpha
end

# Save trace of Markov chain
writecsv("./Results/Trace.csv", [trace])

# Output results and graphs
println("Posterior median beta_0: ", median(trace[burn_in+1:n_iter+1,1]))
println("Posterior median beta_1: ", median(trace[burn_in+1:n_iter+1,2]))
println("Posterior median r: ", median(trace[burn_in+1:n_iter+1,3]))
println("Acceptance rate: ", mean(acceptRate[burn_in+1:n_iter+1]))

beta_0_tracePlot = plot(trace[1:n_iter+1,1])
savefig(beta_0_tracePlot, "./Figs/beta_0_tracePlot.png")
beta_1_tracePlot = plot(trace[1:n_iter+1,2])
savefig(beta_1_tracePlot, "./Figs/beta_1_tracePlot.png")
r_tracePlot = plot(trace[1:n_iter+1,3])
savefig(r_tracePlot, "./Figs/r_tracePlot.png")

beta_0_hist = histogram(trace[burn_in+1:n_iter+1,1])
savefig(beta_0_hist, "./Figs/beta_0_hist.png")
beta_1_hist = histogram(trace[burn_in+1:n_iter+1,2])
savefig(beta_1_hist, "./Figs/beta_1_hist.png")
r_hist = histogram(trace[burn_in+1:n_iter+1,3])
savefig(r_hist, "./Figs/r_hist.png")

ind95 = convert(Int, floor((n_iter - burn_in)*0.95))
ind05 = convert(Int, floor((n_iter - burn_in)*0.05))
beta_0_sort = sort(trace[burn_in+1:n_iter+1,1])
beta_0_cinf = [beta_0_sort[ind05], beta_0_sort[ind95]]
println("Central 95% beta_0 interval: ", beta_0_cinf)
beta_1_sort = sort(trace[burn_in+1:n_iter+1,2])
beta_1_cinf = [beta_1_sort[ind05], beta_1_sort[ind95]]
println("Central 95% beta_1 interval: ", beta_1_cinf)
r_sort = sort(trace[burn_in+1:n_iter+1,3])
r_cinf = [r_sort[ind05], r_sort[ind95]]
println("Central 95% r interval: ", r_cinf)

# Generate predictions of out-of-sample y
yOOS = 60
predict = sort(nbPostPredict(trace[burn_in + 2:n_iter+1,:], yOOS))

# Display summary statistics for predictions
println("Posterior predictive median y, for x = ", yOOS, ": ", median(predict))

n_predict = length(predict)
ind95 = convert(Int, floor(n_predict * 0.95))
ind05 = convert(Int, floor(n_predict * 0.05))
y_predict_cinf = [predict[ind05], predict[ind95]]
println("Central 95% predictive y interval, for x = ", yOOS , ": ", y_predict_cinf)
yOOS_hist = histogram(predict)
savefig(yOOS_hist, "./Figs/yOOS_hist.png")
