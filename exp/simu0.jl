# simuate N = 64, M = 32 change ρ, to see survival fraction

using LinearAlgebra, DifferentialEquations, Plots, Statistics, JSON3

function ResConLog!(du,u,para,t)
    N_s,N_r,G,C,g,K,δ = para
    du[1:N_s] = u[1:N_s] .* (G*u[N_s+1:N_r+N_s] - δ)
    du[N_s+1:N_r+N_s] = u[N_s+1:N_r+N_s] .* (g.*(K - u[N_s+1:N_r+N_s]) - C'*u[1:N_s]) .+ 1e-9
end

N = 64
M = 32N = 64

# for 20 different ρ, each ρ, sample 10 communities (1 initial condition each community), 200 simulation cases in total

num_ρ = 20
ρ_span = LinRange(0.05,1.0,num_ρ)
num_try = 10

tspan = (0.0, 5000.0) # time for one test solution

results = Dict{Integer, Dict}()

@time begin
    Threads.@threads for i in 1:num_ρ
        ρ = ρ_span[i]
        vals = Dict{Integer, Dict}()
        for j in 1:num_try
            # sample 
            Tgc = randn(N, M , 2)
            G = Tgc[:,:,1]
            C = ρ * G + √(1-ρ^2) * Tgc[:,:,2]

            G = G .+ 10.0
            C = C .+ 10.0

            Ss = 0.01 .+ 0.99*rand(N)
            Rs = 0.01 .+ 0.99*rand(M)

            δ = G * Rs
            g = (0.1 .+ 0.9*rand(M)) * 1
            K = C' * Ss ./ g + Rs
            # end of sample

            para = (N,M,G,C,g,K,δ)

            u0 = zeros(N+M)
            u0[1:N] = Ss .* (1 .+ 1.0*(rand(N) .- 0.5))
            u0[N+1:end] = Rs .* (1 .+ 1.0*(rand(M) .- 0.5))

            prob = ODEProblem(ResConLog!,u0,tspan,para)
            #VCABM3() or Tsit5() or TRBDF2(), AutoVern7(Rodas4())
            sol = solve(prob, VCABM3(), saveat = 10, isoutofdomain = (u,p,t)->any(x->x < 0.0,u))
            Send = mean(sol.u[end-50:end])[1:N]
            SF = length(Send[Send .> 1e-7]) / N # survival fraction

            val = Dict(
            "rho" => ρ,
            "SF" => SF)
            vals[j] = val
        end
        results[i] = vals
    end
end

# saving
filename = "../data/simu0.json"
open(filename, "w") do io #save file
    JSON3.pretty(io, results)
end