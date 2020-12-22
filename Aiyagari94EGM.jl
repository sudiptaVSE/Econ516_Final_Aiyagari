using LinearAlgebra
using Parameters
using IterativeSolvers
using FastGaussQuadrature
using ForwardDiff
using QuantEcon
using Plots
using Arpack
using BenchmarkTools
#include("utils.jl")
################################### Model types #########################

uPrime(c,γ) = c.^(-γ)
uPrimeInv(up,γ) = up.^(-1.0/γ)

struct AiyagariParametersEGM{T <: Real}
    β::T
    α::T
    δ::T
    γ::T
    ρ::T
    σz::T #st. deviation of Z shock
    σ::T #job separation
    lamw::T #job finding prob
    Lbar::T
    amin::T
    error::T
end
struct AiyagariModelEGM{T <: Real,I <: Integer}
    params::AiyagariParametersEGM{T}
    aGrid::Array{T,1} ##Policy grid
    aGridl::Array{T,1}
    na::I ##number of grid points in policy function
    dGrid::Array{T,1}
    nd::I ##number of grid points in distribution
    states::Array{T,1} ##earning states 
    ns::I ##number of states
    Trans_mat::Array{T,2}
end
mutable struct AggVarsEGM{S <: Real,T <: Real}
    R::S
    w::T
end

function PricesEGM(K,Z,params::AiyagariParametersEGM)
    @unpack β,α,δ,γ,ρ,σ,lamw,Lbar = params
    R = Z*α*(K/Lbar)^(α-1.0) + 1.0 - δ
    w = Z*(1.0-α)*(K/Lbar)^(1.0-α) 
    
    return AggVarsEGM(R,w)
end

function AiyagariEGM(
    K::T,
    β::T = 0.98,
    α::T = 0.4,
    δ::T = 0.02,
    γ::T = 2.0,
    ρ::T = 0.95,
    σz::T = 1.0,
    σ::T = 0.2,
    lamw::T = 0.6,
    Lbar::T = 1.0,
    amin::T = 1e-9,
    amax::T = 200.0,
    error::T = 1000000000.0,
    na::I = 201,
    nd::I = 201,
    ns::I = 2,
    endow = [1.0;2.5]) where{T <: Real,I <: Integer}

    #############Params
    params = AiyagariParametersEGM(β,α,δ,γ,ρ,σz,σ,lamw,Lbar,amin,error)
    AggVars = PricesEGM(K,1.0,params)
    @unpack R,w = AggVars

    ################## Policy grid
    function make_grid(a_min,a_max,na, s)
        x = range(a_min,step=0.5,length=na)
        grid = a_min .+ (a_max-a_min)*(x.^s/maximum(x.^s))
        return grid
    end
    aGrid = make_grid(amin,amax,na,4.0)
    ################### Distribution grid
    #dGrid = collect(range(aGrid[1],stop = aGrid[end],length = nd))
    #aGrid = collect(range(amin,stop = amax,length = na))
    dGrid=aGrid

    ################## Transition
    Trans_mat = [1.0-lamw σ ;lamw 1.0-σ]
    dis = LinearAlgebra.eigen(Trans_mat)
    m = argmin(abs.(dis.values .- 1.0)) 
    stdist = abs.(dis.vectors[:,m]) / sum(abs.(dis.vectors[:,m]))
    lbar = dot(stdist,endow)
    states = endow/lbar

    @assert sum(Trans_mat[:,1]) == 1.0 ###sum to 1 across rows
    #summing across rows is nice as we don't need to transpose transition before taking eigenvalue
    
    guess = vcat(10.0 .+ aGrid,10.0 .+ aGrid)

    
    return AiyagariModelEGM(params,aGrid,vcat(aGrid,aGrid),na,dGrid,nd,states,ns,Trans_mat),guess,AggVars
end

function interpEGM(policy::AbstractArray,
                grid::AbstractArray,
                x::T,
                na::Integer) where{T <: Real}
    np = searchsortedlast(policy,x)

    ##Adjust indices if assets fall out of bounds
    (np > 0 && np < na) ? np = np : 
        (np == na) ? np = na-1 : 
            np = 1        
    #@show np
    ap_l,ap_h = policy[np],policy[np+1]
    a_l,a_h = grid[np], grid[np+1] 
    ap = a_l + (a_h-a_l)/(ap_h-ap_l)*(x-ap_l) 
    
    above =  ap > 0.0 
    return above*ap,np
end

function get_consEGM(policy::AbstractArray,
               Aggs::AggVarsEGM,
               CurrentAssets::AbstractArray,
               AiyagariModel::AiyagariModelEGM,
               cpolicy::AbstractArray) 
    
    @unpack aGrid,na,ns,states = AiyagariModel
    policy = reshape(policy,na,ns)
    for si = 1:ns
        for ai = 1:na
            asi = (si - 1)*na + ai
            cpolicy[asi] = Aggs.R*currentassets[asi] + Aggs.w*states[si] - interpEGM(policy[:,si],aGrid,currentassets[asi],na)[1]
        end
    end
    return cpolicy
end

function EulerBackEGM(policy::AbstractArray,
                   Aggs::AggVarsEGM,
                   Aggs_P::AggVarsEGM,
                   AiyagariModel::AiyagariModelEGM,
                   cpolicy::AbstractArray,
                   apolicy::AbstractArray)
    
    @unpack params,na,nd,ns,aGridl,Trans_mat,states = AiyagariModel
    @unpack γ,β = params

    R_P,w_P = Aggs_P.R,Aggs_P.w
    R,w = Aggs.R,Aggs.w
    
    cp = get_consEGM(policy,Aggs_P,aGridl,AiyagariModel,cpolicy)
    upcp = uPrime(cp,γ)
    Eupcp = copy(cpolicy)
    #Eupcp_sp = 0.0

    for ai = 1:na
        for si = 1:ns
            asi = (si-1)*na + ai
            Eupcp_sp = 0.0
            for spi = 1:ns
                aspi = (spi-1)*na + ai
                Eupcp_sp += Trans_mat[spi,si]*upcp[aspi]
            end
            Eupcp[asi] = Eupcp_sp 
        end
    end

    upc = R_P*β*Eupcp

    c = uPrimeInv(upc,γ)

    for ai = 1:na
        for si = 1:ns
            asi = (si-1)*na+ai
            apolicy[asi] = (aGridl[asi] + c[asi] - w*states[si])/R
        end
    end

    return apolicy,c
end


function SolveEGM(policy::AbstractArray,
                  Aggs::AggVarsEGM,
                  AiyagariModel::AiyagariModelEGM,
                  cpolicy::AbstractArray,
                  apolicy::AbstractArray,tol = 1e-17)
    @unpack ns,na = AiyagariModel

    for i = 1:10000
        a = EulerBackEGM(policy,Aggs,Aggs,AiyagariModel,cpolicy,apolicy)[1]
        if (i-1) % 50 == 0
            test = abs.(a - policy)/(abs.(a) + abs.(policy))
            #println("iteration: ",i," ",maximum(test))
            if maximum(test) < tol
                println("Solved in ",i," ","iterations")
                break
            end
        end
        policy = copy(a)
    end
    return policy
end

function MakeTransMatEGM(policy,AiyagariModel,tmat)
    @unpack ns,na,aGrid,Trans_mat = AiyagariModel
    policy = reshape(policy,na,ns)
    for a_i = 1:na
        for j = 1:ns
            x,i = interpEGM(policy[:,j],aGrid,aGrid[a_i],na)
            p = (aGrid[a_i] - policy[i,j])/(policy[i+1,j] - policy[i,j])
            p = min(max(p,0.0),1.0)
            sj = (j-1)*na
            for k = 1:ns
                sk = (k-1)*na
                tmat[sk+i+1,sj+a_i] = p * Trans_mat[k,j]
                tmat[sk+i,sj+a_i] = (1.0-p) * Trans_mat[k,j]
            end
        end
    end
    return tmat
end


function MakeTransEGM(AiyagariModel)
    @unpack params,nd,Trans_mat = AiyagariModel
    δ = params.δ
    eye = LinearAlgebra.eye(eltype(Trans_mat),nd)
    hcat(vcat(eye*Trans_mat[1,1],eye*Trans_mat[2,1]),vcat(eye*Trans_mat[1,2],eye*Trans_mat[2,2]))
end

function StationaryDistributionEGM(T,AiyagariModel)
    @unpack ns,nd = AiyagariModel 
    λ, x = powm!(T, rand(ns*nd), maxiter = 100000,tol = 1e-15)
    return x/sum(x)
end

function equilibriumEGM(
    initialpolicy::AbstractArray,
    AiyagariModel::AiyagariModelEGM,
    K0::T,
    tol = 1e-10,maxn = 50)where{T <: Real}

    @unpack params,aGrid,na,dGrid,nd,ns = AiyagariModel

    EA = 0.0

    tmat = zeros(eltype(initialpolicy),(na*ns,na*ns))
    cmat = zeros(eltype(initialpolicy),na*ns)

    ###Start Bisection
    uK,lK = K0, 0.0
    
    policy = initialpolicy
    #uir,lir = initialR, 1.0001
    print("Iterate on aggregate assets")
    for kit = 1:maxn
        Aggs = PricesEGM(K0,1.0,params) ##steady state
        cmat .= 0.0
        policy = SolveEGM(policy,Aggs,AiyagariModel,cmat,cmat)

        #Stationary transition
        tmat .= 0.0
        trans = MakeTransMatEGM(pol,AiyagariModel,tmat)
        D = StationaryDistributionEGM(trans,AiyagariModel)

        #Aggregate savings
        EA = dot(vcat(dGrid,dGrid),D)
        
        if (EA > K0) ### too little lending -> low r -> too much borrowing 
            uK = min(EA,uK)  
            lK = max(K0,lK)
            K0 = 1.0/2.0*(lK + uK)
        else ## too much lending -> high r -> too little borrowing
            uK = min(K0,uK)
            lK = max(EA,lK)
            K0 = 1.0/2.0*(lK + uK)
        end
        println("Interest rate: ",Aggs.R," ","Bond Supply: ",EA)
        #@show K0
        if abs(EA - K0) < 1e-7
            println("Markets clear!")
            #println("Interest rate: ",R," ","Bonds: ",EA)
            cmat .= 0.0
            policyA,policyC = EulerBackEGM(policy,Aggs,Aggs,AiyagariModel,cmat,cmat)
            return policyA,policyC,D,EA,Aggs
            break
        end
    end
    
    return println("Markets did not clear")
end

function EulerResidualEGM(policy::AbstractArray,
                       pol_P::AbstractArray,
                       Aggs::AggVarsEGM,
                       Aggs_P::AggVarsEGM,
                       AiyagariModel::AiyagariModelEGM,
                       cmat::AbstractArray,
                          amat::AbstractArray)
    
    a,c = EulerBackEGM(pol_P,Aggs,Aggs_P,AiyagariModel,cmat,amat)
    c2 = get_consEGM(policy,Aggs,a,AiyagariModel,cmat)

    return (c ./ c2 .- 1.0)
end


function WealthResidualEGM(policy::AbstractArray,
                        D_L::AbstractArray,
                        D::AbstractArray,
                        AiyagariModel::AiyagariModelEGM,
                        tmat::AbstractArray)
    return (D - MakeTransMatEGM(policy,AiyagariModel,tmat) * D_L)[2:end]
end

function AggResidual(D::AbstractArray,K,Z_L,Z,epsilon::AbstractArray,AiyagariModel::AiyagariModelEGM)
    @unpack params,dGrid = AiyagariModel
    @unpack ρ,σz = params
    ϵz = epsilon[1]

    AggAssets = dot(D,vcat(dGrid,dGrid))
    AggEqs = vcat(
        AggAssets - K, #bond market clearing
        log(Z) - ρ*log(Z_L) - ϵz, #TFP evol
    ) 
    
    return AggEqs
end

function FEGM(X_L::AbstractArray,
           X::AbstractArray,
           X_P::AbstractArray,
           epsilon::AbstractArray,
           AiyagariModel::AiyagariModelEGM,
           pos)
    
    @unpack params,na,ns,nd,dGrid = AiyagariModel

    
    m = na*ns
    md = nd*ns
    pol_L,D_L,Agg_L = X_L[1:m],X_L[m+1:m+md-1],X_L[m+md:end]
    policy,D,Agg = X[1:m],X[m+1:m+md-1],X[m+md:end]
    pol_P,D_P,Agg_P = X_P[1:m],X_P[m+1:m+md-1],X_P[m+md:end]

    K_L,Z_L = Agg_L
    K,Z = Agg
    K_P,Z_P = Agg_P

    D_L = vcat(1.0-sum(D_L),D_L)
    D   = vcat(1.0-sum(D),D)
    D_P = vcat(1.0-sum(D_P),D_P)
    
    
    Price = PricesEGM(K_L,Z,params)
    Price_P = PricesEGM(K,Z_P,params)
    #@show Price_P
    #Need matrices that pass through intermediate functions to have the same type as the
    #argument of the derivative that will be a dual number when using forward diff. In other words,
    #when taking derivative with respect to X_P, EE, his, his_rhs must have the same type as X_P
    if pos == 1 
        cmat = zeros(eltype(X_L),na*ns)
        cmat2 = zeros(eltype(X_L),na*ns)
        tmat = zeros(eltype(X_L),(ns*na,ns*na))
    elseif pos == 2
        cmat = zeros(eltype(X),na*ns)
        cmat2 = zeros(eltype(X),na*ns)
        tmat = zeros(eltype(X),(ns*na,ns*na))
    else
        cmat = zeros(eltype(X_P),na*ns)
        cmat2 = zeros(eltype(X_P),na*ns)
        tmat = zeros(eltype(X_P),(ns*na,ns*na))
    end
    agg_root = AggResidual(D,K,Z_L,Z,epsilon,AiyagariModel)
    dist_root = WealthResidualEGM(policy,D_L,D,AiyagariModel,tmat) ###Price issue
    euler_root = EulerResidualEGM(policy,pol_P,Price,Price_P,AiyagariModel,cmat,cmat2)

    return vcat(euler_root,dist_root,agg_root)
end

function EulerBackError(policy::AbstractArray,
                   Aggs::AggVarsEGM,
                   Aggs_P::AggVarsEGM,
                   AiyagariModel::AiyagariModelEGM,
                   cpolicy::AbstractArray,
                   apolicy::AbstractArray,
                   Grid::AbstractArray)
    
    @unpack params,na,nd,ns,aGridl,states,Trans_mat = AiyagariModel
    @unpack γ,β = params

    R_P,w_P = Aggs_P.R,Aggs_P.w
    R,w = Aggs.R,Aggs.w
    ng = div(length(Grid),ns)
    cp = get_c_error(policy,Aggs_P,Grid,AiyagariModel,cpolicy)
    upcp = uPrime(cp,γ)
    Eupcp = copy(cpolicy)

    for ai = 1:ng
        for si = 1:ns
            asi = (si-1)*ng + ai
            Eupcp_sp = 0.0
            for spi = 1:ns
                aspi = (spi-1)*ng + ai
                Eupcp_sp += Trans_mat[spi,si]*upcp[aspi]
            end
            Eupcp[asi] = Eupcp_sp 
        end
    end

    upc = R_P*β*Eupcp

    c = uPrimeInv(upc,γ)

    for ai = 1:ng
        for si = 1:ns
            asi = (si-1)*ng+ai
            apolicy[asi] = (Grid[asi] + c[asi] - w*states[si])/R
        end
    end

    return apolicy,c
end
function get_cons_error(policy::AbstractArray,
               Aggs::AggVarsEGM,
               currentassets::AbstractArray,
               AiyagariModel::AiyagariModelEGM,
               cpolicy::AbstractArray) 
    
    @unpack aGrid,na,ns,states = AiyagariModel
    
    policy = reshape(policy,na,ns)
    Gsize = div(length(currentassets),ns)
    for si = 1:ns
        for ai = 1:Gsize
            asi = (si - 1)*Gsize + ai
            cpolicy[asi] = Aggs.R*currentassets[asi] + Aggs.w*states[si] - interpEGM(policy[:,si],aGrid,currentassets[asi],na)[1]
        end
    end
    return cpolicy
end

function EulerResidualError(policy::AbstractArray,
                       pol_P::AbstractArray,
                       Aggs::AggVarsEGM,
                       Aggs_P::AggVarsEGM,
                       AiyagariModel::AiyagariModelEGM,
                       cmat::AbstractArray,
                       amat::AbstractArray,
                       Grid::AbstractArray)    
    
    a,c = EulerBackError(pol_P,Aggs,Aggs_P,AiyagariModel,cmat,amat,Grid)
    c2 = get_c_error(policy,Aggs,a,AiyagariModel,cmat)
    return (c ./ c2 .- 1.0)
end
