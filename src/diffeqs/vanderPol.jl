mutable struct vanderPol <: SpecialDiffSys
    μ::Float64
    noise::AbstractArray
end

name(::vanderPol) = "van der Pol equation"
dimension(::vanderPol) = 2

vanderPol(; μ=0.3, noise=zeros(2)) = vanderPol(μ, noise)

function vec_field(vdp::vanderPol, x::AbstractArray)
    u, v = x
    μ = vdp.μ
    du = v
    dv = -u-μ*v*(u^2-1.0)
    return [du, dv]
end

function jacobian(vdp::vanderPol, x::AbstractArray)
    u, v = x
    μ = vdp.μ
    J = [0 1; -1-2μ*u*v -μ*(u^2-1)]
    return J
end

