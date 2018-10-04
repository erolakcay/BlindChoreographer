using DifferentialEquations
using StatsBase

# Variables
counter = 1;
sims = 100;
time = 1000;
Output = zeros(4200,4);

for alpha = 4:8:12 # Refinement of the dollar
    # Pi, the payoff matrix #
    Pi = zeros(alpha+2,alpha+2);
    for m = 0:alpha
        for n = 0:alpha
            if m + n <= alpha
                Pi[m+1,n+1] = m;
            end
        end
    end
    Pi = Pi/alpha;
    Pi[alpha+2,:] = ones(1,alpha+2)/alpha;
    Pi[2:alpha,alpha+2] = ones(alpha-1,1)/alpha;
    Pi[alpha+2,alpha+1] = 0;
    ###############
    for v = 5:25 # Number of vertices
    	# Construct ring with K neighbours
        A = zeros(v,v);
        K = 2; # twice the mean degree (even number)
        for m = 1:v
            for n = 1:K
                A[m,mod(m+n-1,v)+1] = 1;
            end
        end
        A = A + transpose(A);
        ###############
        for s = 1:sims
            # Variables #
            P = (alpha+1)*ones(Int16,v,1); # Prescriptions for each label
            L = ones(Int16,v,1); # Labels for each vertex for each norm
            x = ones(1); # Frequency of each norm
            numNorms = 1;
            Payoff = zeros(2,1); # Payoff matrix for norms
            ###############
            for t = 1:time # Run simulations
                # Mutation #
                norm = rand(1:numNorms); # Choose the norm to be mutated
                mutP = P[:,norm];
                mutL = L[:,norm];
                if rand() < 0.5 # The probability of mutating the prescription for a label
                    label = sample(mutL); # Select a label applied to a vertex
                    while P[:,norm] == mutP
                        mutP[label] = rand(0:alpha+1); # Prescribe a new strategy for that label
                    end
                elseif (any(y->y!=mutL[1],mutL)) && (rand() < 0.5) # Mutate the label at a vertex
                    mutvert = rand(1:v);
                    mutL[mutvert] = sample(mutL[mutL.!=mutL[mutvert]]); # Set a random vertex to a new label
                else
                    mutL[rand(1:v)] = rand(1:v);
                end
                ###############

                # Construct payoff matrix
                tempP = [P mutP];
                tempL = [L mutL];
                Payoff = zeros(numNorms+1,numNorms+1);
                for m = 1:numNorms+1
                    for n = 1:numNorms+1
                        for p = 1:v
                            for q = 1:v
                                if A[p,q] == 1
                                    Payoff[m,n] += Pi[tempP[tempL[p,m],m]+1,tempP[tempL[q,n],n]+1];
                                end
                            end
                        end
                    end
                end
                Payoff = (Payoff/sum(A));
                ###############

                # Check invasability: E(mut,pop) > E(pop,pop), E(mut,mut) >= E(pop,mut)
                # If E(mut,mut) = E(pop,mut), then replace resident(s) with mutant
                Fit_mutVpop = dot(vec(Payoff[numNorms+1,1:numNorms]),x);
                Fit_popVpop = dot(Payoff[1:numNorms,1:numNorms]*x,x);
                if Fit_mutVpop < Fit_popVpop
                    Payoff = Payoff[1:end-1,1:end-1];
                    continue
                elseif Fit_mutVpop == Fit_popVpop
                    Fit_popVmut = dot(Payoff[1:numNorms,numNorms+1],x);
                    if Payoff[numNorms+1,numNorms+1] < Fit_popVmut
                        Payoff = Payoff[1:end-1,1:end-1];
                        continue
                    elseif Payoff[numNorms+1,numNorms+1] == Fit_popVmut
                        P[:,norm] = mutP;
                        L[:,norm] = mutL;
                        Payoff = Payoff[setdiff(1:end,norm),:];
                        Payoff = Payoff[:,setdiff(1:end,norm)];
                        continue
                    end
                end
                ###############

                # If mutant has invaded, find new equilibrium
                u0 = [x; min(x[norm],0.05)];
                u0[norm] += -min(x[norm],0.05);
                tspan = (0.0,100.0);
                f(u,p,t) = u.*Payoff*u - u*dot(Payoff*u,u);
                problem = ODEProblem(f,u0,tspan);
                sol = solve(problem)
                x = sol[:,end];
                P = [P mutP][:,vec(x).>0.05];
                L = [L mutL][:,vec(x).>0.05];
                Payoff = Payoff[:,vec(x).>0.05];
                Payoff = Payoff[vec(x).>0.05,:];
                x = x[x.>0.05];
                x = x/sum(x);
                numNorms = length(x);
            end
            #Q = P[L];
            #X = kron(x',ones(v,1));
            #X = X[Q.!=alpha+1];
            #Q = Q[Q.!=alpha+1]/alpha;
            # var(Q,weights(X))
            y = [1:v;];
            mono = 1;
            for m = 1:numNorms
                for n = (m+1):numNorms
                    if P[L[y,m],m] != P[L[y,n],n]
                        mono = 0;
                    end
                end
            end
            Output[counter,:] = [v dot(Payoff*x,x) alpha mono];
            counter += 1;
        end # sims
    end # v
end # alpha
writedlm("ringK2.txt", Output)
