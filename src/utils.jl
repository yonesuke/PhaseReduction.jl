abstract type AbstractDiffSys end

abstract type SpecialDiffSys end

function Base.show(io::IO, ::MIME"text/plain", sys::SpecialDiffSys)
    println(io, typeof(sys))
    for fname in fieldnames(typeof(sys))
        fval = getfield(sys, fname)
        println(io, "    $fname = $fval")
    end
end

mutable struct DiffSys <: AbstractDiffSys
    n_dim::Int64
    dt::Float64
    t_max::Float64
    func::Function
    noise::AbstractArray
    name::String
end

function Base.show(io::IO, ::MIME"text/plain", sys::DiffSys)
    println(io, typeof(sys))
    println(io, "    Differential equation name: ", sys.name)
    println(io, "    Differential equation dimension: ", sys.n_dim)
    println(io, "    time step: ", sys.dt)
    println(io, "    Calculation time: ", sys.t_max)
    println(io, "    noise: ", sys.noise)
end

# DiffSys(sys::SpecialDiffSys; dt=0.01, t_max=200) = error("unimplemented")
function DiffSys(sys::SpecialDiffSys; dt=0.01, t_max=200)
    n_dim = dimension(sys)
    func = x -> vec_field(sys, x)
    noise = sys.noise
    nm = name(sys)
    return DiffSys(n_dim, dt, t_max, func, noise, nm)
end
DiffSys(sys::DiffSys) = sys

function EulerMaruyama(func::Function, noise::AbstractArray, dt::Float64, x0::AbstractArray)
    return x0 + dt*func(x0) + √(dt)*noise.*randn(length(noise))
end

function RungeKutta(func::Function, dt::Float64, x0::AbstractArray)
    k1 = func(x0)
    k2 = func(x0+0.5dt*k1)
    k3 = func(x0+0.5dt*k2)
    k4 = func(x0+dt*k3)
    dx = dt*(k1+2k2+2k3+k4)/6
    return x0+dx
end

EulerMaruyama(sys::DiffSys, x0::AbstractArray) = EulerMaruyama(sys.func, sys.noise, sys.dt, x0)

function integrate(sys::DiffSys, x0::AbstractArray)
    ts = [0:sys.dt:sys.t_max;]
    n_step = length(ts)
    n_dim = sys.n_dim
    orbits = zeros(n_step, n_dim)
    x = copy(x0)
    orbits[1, :] = x
    for i in 2:n_step
        x = EulerMaruyama(sys, x)
        orbits[i, :] = x
    end
    return ts, orbits
end

integrate(sys::SpecialDiffSys, x0::AbstractArray; dt=0.01, t_max=200) = integrate(DiffSys(sys, dt=dt, t_max=t_max), x0)

mutable struct CoupledDiffSys <: AbstractDiffSys
    n_body::Int64
    dt::Float64
    t_max::Float64
    SelfDiffSyss::Array{DiffSys}
    OriginalSpecialDiffSyss::Array{SpecialDiffSys}
    CouplingFuncDict::Dict
end

function Base.show(io::IO, ::MIME"text/plain", sys::CoupledDiffSys)
    println(io, typeof(sys))
    println(io, "    Differential equation number of body: ", sys.n_body)
    total_dim = sum([s.n_dim for s in sys.SelfDiffSyss])
    println(io, "    Differential equation dimension: ", total_dim)
    println(io, "    time step: ", sys.dt)
    println(io, "    Calculation time: ", sys.t_max)
    println(io, "    Name of each systems")
    for i in 1:sys.n_body
        println(io, "        Number $i: $(sys.SelfDiffSyss[i].name)")
    end
    couplings = keys(sys.CouplingFuncDict)
    println(io, "    Coupled through")
    for (i,j) in couplings
        println(io, "        $i -> $j")
    end
end

function CoupledDiffSys(OriginalSpecialDiffSyss::Array{SpecialDiffSys}, CouplingFuncDict::Dict; dt=0.01, t_max=200)
    n_body = length(OriginalSpecialDiffSyss)
    SelfDiffSyss = [DiffSys(sys) for sys in OriginalSpecialDiffSyss]
    return CoupledDiffSys(n_body, dt, t_max, SelfDiffSyss, OriginalSpecialDiffSyss, CouplingFuncDict)
end

function ith_coupling_idxs(CouplingFuncDict::Dict, i::Int64)
    idxs = [couplings[2] for couplings in collect(keys(CouplingFuncDict)) if couplings[1]==i]
    return idxs
end

function create_vec(sys::CoupledDiffSys)
    dims = [s.n_dim for s in sys.SelfDiffSyss]
    cumdims = cumsum(dims)
    self_vecs = [s.func for s in sys.SelfDiffSyss]
    func = x -> begin
        xs = [x[1:cumdims[1]]]
        for i in 2:sys.n_body
            push!(xs, x[cumdims[i-1]+1:cumdims[i]])
        end
        vs = []
        for i in 1:sys.n_body
            v = self_vecs[i](xs[i])
            coupling_idxs = ith_coupling_idxs(sys.CouplingFuncDict, i)
            for j in coupling_idxs
                v += sys.CouplingFuncDict[(i,j)](xs[i], xs[j])
            end
            push!(vs, v)
        end
        return vcat(vs...)
    end
    return func
end

function DiffSys(sys::CoupledDiffSys; dt=0.01, t_max=200)
    dims = [s.n_dim for s in sys.SelfDiffSyss]
    n_dim = sum(dims)
    func = create_vec(sys)
    noises = vcat([s.noise for s in sys.SelfDiffSyss]...)
    names = ""
    return DiffSys(n_dim, dt, t_max, func, noises, names)
end

integrate(sys::CoupledDiffSys, x0::AbstractArray; dt=0.01, t_max=200) = integrate(DiffSys(sys, dt=dt, t_max=t_max), x0)