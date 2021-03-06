using Plots

include("../src/model.jl")
include("../src/datasets.jl")

data = gen_synthetic(N=500, T=2000)

plot(xlabel="Time", ylabel="Loss")

alg_results = Dict()
for (alg, options, label) in [
    [:hals, Dict(), "HALS"],
    #[:mult, Dict(), "MULT"],
    [:anls, Dict(), "ANLS"],
]
    results = fit_cnmf(data, L=10, K=5,
                       alg=alg, alg_options=options,
                       max_itr=1000, max_time=10)

    plot!(results.time_hist, results.loss_hist, label=label)
    scatter!(results.time_hist, results.loss_hist, markersize=1, label="")

    alg_results[alg] = results
end

savefig("cnmf_alg_comparison.png")
gui()
;
