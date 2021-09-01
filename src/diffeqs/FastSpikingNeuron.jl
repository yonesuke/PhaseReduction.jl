mutable struct FastSpikingNeuron <: SpecialDiffSys
    C::Float64
    G_Na::Float64
    G_K::Float64
    G_L::Float64
    E_Na::Float64
    E_K::Float64
    E_L::Float64
    I_bias::Float64
    noise::AbstractArray
end

name(::FastSpikingNeuron) = "Fast spiking neuron equation"
dimension(::FastSpikingNeuron) = 4

FastSpikingNeuron(; C=1.0, G_Na=112.0, G_K=224.0, G_L=0.1, E_Na=55.0, E_K=-97.0, E_L=-70.0, I_bias=6.0, noise=zeros(4)) = FastSpikingNeuron(C, G_Na, G_K, G_L, E_Na, E_K, E_L, I_bias, noise)

function vec_field(fsn::FastSpikingNeuron, x::AbstractArray)
    V, m, h, n = x
    C = fsn.C
    G_Na, G_K, G_L = fsn.G_Na, fsn.G_K, fsn.G_L
    E_Na, E_K, E_L = fsn.E_Na, fsn.E_K, fsn.E_L
    I_bias = fsn.I_bias
    _dV = G_Na*m^3*h*(E_Na-V) + G_K*n^2*(E_K-V) + G_L*(E_L-V) + I_bias
    dV = _dV/C
    αₘ = 40(V-75)/(1-exp((75-V)/13.5))
    βₘ = 1.2262exp(-V/42.248)
    dm = αₘ*(1-m) - βₘ*m
    αₕ = 0.0035exp(-V/24.186)
    βₕ = 0.017(-51.25-V)/(exp((-51.25-V)/5.2)-1)
    dh = αₕ*(1-h) - βₕ*h
    αₙ = (V-95)/(1-exp((95-V)/11.8))
    βₙ = 0.025exp(-V/22.222)
    dn = αₙ*(1-n) - βₙ*n
    return [dV, dm, dh, dn]
end

function jacobian(fsn::FastSpikingNeuron, x::AbstractArray)
    V, m, h, n = x
    C = fsn.C
    G_Na, G_K, G_L = fsn.G_Na, fsn.G_K, fsn.G_L
    E_Na, E_K = fsn.E_Na, fsn.E_K
    _dVdV = -G_Na*m^3*h - G_K*n^2 - G_L
    _dVdm = 3m^2*G_Na*h*(E_Na-V)
    _dVdh = G_Na*m^3*(E_Na-V)
    _dVdn = 2G_K*n*(E_K-V)
    dVs = [_dVdV _dVdm _dVdh _dVdn]/C
    αₘ = 40(V-75)/(1-exp((75-V)/13.5))
    βₘ = 1.2262exp(-V/42.248)
    dαₘ = begin
        y = (V-75)/13.5
        40*(1-exp(-y)-y*exp(-y))/(1-exp(-y))^2
    end
    dβₘ = -(1.2262/42.248)*exp(-V/42.248)
    dms = [dαₘ*(1-m)-dβₘ*m -αₘ-βₘ 0 0]
    αₕ = 0.0035exp(-V/24.186)
    βₕ = 0.017(-51.25-V)/(exp((-51.25-V)/5.2)-1)
    dαₕ = -(0.0035/24.186)*exp(-V/24.186)
    dβₕ = begin
        y = (V+51.25)/5.2
        0.017*(1-exp(-y)-y*exp(-y))/(1-exp(-y))^2
    end
    dhs = [dαₕ*(1-h)-dβₕ*h 0 -αₕ-βₕ 0]
    αₙ = (V-95)/(1-exp((95-V)/11.8))
    βₙ = 0.025exp(-V/22.222)
    dαₙ = begin
        y = (V-95)/11.8
        (1-exp(-y)-y*exp(-y))/(1-exp(-y))^2
    end
    dβₙ = -(0.025/22.222)*exp(-V/22.222)
    dns = [dαₙ*(1-n)-dβₙ*n 0 0 -αₙ-βₙ]
    return vcat(dVs, dms, dhs, dns)
end