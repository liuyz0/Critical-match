# attractor basin sizes

# distributed
using Distributed, MAT, ClusterManagers
#addprocs(4) # add 4 cores, different on cluster!
addprocs(SlurmManager(parse(Int,ENV["SLURM_NTASKS"])-1))
println("Added workers: ", nworkers())
flush(stdout)

@everywhere begin
    using LinearAlgebra, Statistics, Distributions, DynamicalSystems, DifferentialEquations

    function ResConLin!(du,u,para,t)
        N_s,N_r,G,C,l,kappa,δ = para
        du[1:N_s] = u[1:N_s].*(G*u[N_s+1:N_r+N_s] - δ) .+ 1e-7
        du[N_s+1:N_r+N_s] = l.*(kappa - u[N_s+1:N_r+N_s]) - u[N_s+1:N_r+N_s].*(C'*u[1:N_s])
    end

    function sample(N, M, ρ)
        Tgc = randn(N, M, 2)

        ρp = 2 * sin(ρ * π/6)
        G = Tgc[:,:,1]
        C = ρp * G + √(1-ρp^2) * Tgc[:,:,2]

        G = cdf.(Normal(), G)
        C = cdf.(Normal(), C)

        Ss = (0.01 .+ 0.99*rand(N)) * M / N
        Rs = 0.01 .+ 0.99*rand(M)

        l = 0.1 .+ 0.9*rand(M)
        δ = G*Rs
        κ = Rs.*(C' * Ss)./l + Rs
        return Ss, Rs, G, C, l, κ, δ
    end

    function solRCM(Ss,Rs,para,RCM,disturb=1.99,tEnd=5000.0, MIN_POP = 1e-9)
        Ns, Nr = para
        tspan = (0.0, tEnd)
        
        alg = AutoVern7(Rodas4())

        # checks if species fall below extinction threshold
        condition(u, t, integrator) = any(u .< MIN_POP)
        # sets species below extinction threshold to 0
        function affect!(integrator)
            integrator.u[integrator.u.<MIN_POP] .= MIN_POP
        end
        cb = DiscreteCallback(condition, affect!)
        
        # sample an initial condition
        u0 = zeros(Ns+Nr)
        u0[1:Ns] = Ss .* (1 .+ disturb*(rand(Ns) .- 0.5))
        u0[Ns+1:end] = Rs .* (1 .+ disturb*(rand(Nr) .- 0.5))
        iter = 1

        #initialize output
        output = zeros(4)

        stop = 0
        while stop == 0
            prob = ODEProblem(RCM,u0,tspan,para)
            sol = solve(prob, saveat = 10.0, alg, callback = cb)
            meanS = mean(sol.u[end-100:end])[1:Ns]
            index = findall(meanS .> 1e-5)
            flagS = mean(abs.((sol.u[end][index] - meanS[index])./meanS[index]))
            if iter == 15 || flagS <= 1e-3
                stop = 1
                output[1] = length(index)/Ns # fraction of survial
            end
            iter = iter + 1
            u0 = sol.u[end]
            #u0[u0 .< 1.0e-9] .= 1.0e-9
        end

        diffeq = (alg = TRBDF2(), isoutofdomain = (u,p,t)->any(x->x < 0.0,u))
        LyExp = lyapunov(ContinuousDynamicalSystem(RCM,u0,para; diffeq), 5000.0)

        fluc = 0.0
        if iter == 16
            fluc = 1.0
        end
        
        choscriterion = 0.002

        if fluc == 1.0
            if LyExp > choscriterion
                output[2] = 1.0 # chaos
            else
                output[3] = 1.0 # limit cycles
            end
        else
            output[4] = 1.0 # stable
        end

        return output, u0
    end

    function Stability(para,Ss,Rs)
        Ns,Nr,G,C,l,kappa,δ = para
        Js = zeros(Ns+Nr,Ns+Nr)
        Js[1:Ns,Ns+1:Ns+Nr] = Ss .* G
        Js[Ns+1:Ns+Nr,1:Ns] = - Rs .* C'
        Js[Ns+1:Ns+Nr,Ns+1:Ns+Nr] = - diagm(C' * Ss + l)

        Eig_J = real(eigvals(Js))
        stable = false
        if length(Eig_J[Eig_J .>= 1.0e-6]) == 0
            stable = true
        end
        return stable
    end
end


N = 48
ρ = 0.8
M_span = 1:2:128
num_samp = 100
num_init = 50 # important!

idxs = [(i,j) for i=1:length(M_span), j=1:num_samp]
idxs = reshape(idxs,:)

# build parallel function
@everywhere function parallel_sol(ii, idxs, N, M_span, ρ, num_init)
    (i,j) = idxs[ii]
    M = M_span[i]
    Ss, Rs, G, C, l, κ, δ = sample(N,M,ρ)
    
    para = (N,M,G,C,l,κ,δ)

    fstates = zeros(num_init,4)
    if Stability(para, Ss, Rs)
        fstates[:,1] .= 1
    else 
        ufs = Matrix{Float64}(undef, M+N, 0)
        for k in 1:num_init
            fstates[k,:], uf = solRCM(Ss,Rs,para,ResConLin!) # one initial condition
            if fstates[k,4] == 1.0
                ufs = [ufs uf]
            end
        end # end for k 
        if size(ufs)[2] == num_init
            innerprod = ufs' * ufs
            innerprod = diagm(sqrt.(diag(innerprod)).^(-1)) * innerprod * diagm(sqrt.(diag(innerprod)).^(-1))
            PCA = real(eigvals(innerprod))

            if length(PCA[PCA .> 0.05*maximum(PCA)]) <= 1 # single stable
                fstates[:,4] .= 0.0 # alternative stable states
            end
        end
    end

    println("Running to i = ", i, " and j = ",j)
    flush(stdout)

    return fstates
end

@time Fractions_coll = pmap(ii -> parallel_sol(ii, idxs, N, M_span, ρ, num_init), 1:length(idxs))
Fractions = zeros(length(M_span),num_samp,num_init,4)

for ii in eachindex(idxs)
    (i,j) = idxs[ii]
    Fractions[i,j,:,:] = Fractions_coll[ii]
end

# save
myfilename = "./attractor-1.mat"
file = matopen(myfilename, "w")
write(file, "Fractions", Fractions)
close(file)