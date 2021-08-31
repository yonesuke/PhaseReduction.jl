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