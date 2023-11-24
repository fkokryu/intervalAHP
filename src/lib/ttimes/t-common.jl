using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("./optimal-value.jl")

LPResult_t_CommonGround = @NamedTuple{
    # 区間重みベクトル
    wᴸ::Vector{T}, wᵁ::Vector{T},
    W::Vector{Interval{T}}, # ([wᵢᴸ, wᵢᵁ])
    vᴸ::Matrix{T}, vᵁ::Matrix{T},
    v::Vector{Vector{Interval{T}}}, # ([vᵢᴸ, vᵢᵁ])
    optimalValue::T,
    s::Vector{T}
    } where {T <: Real}

function solvetCommonGroundLP(matrices::Vector{Matrix{T}})::LPResult_t_CommonGround{T} where {T <: Real}
    ε = 1e-8 # << 1

    if isempty(matrices)
        throw(ArgumentError("Argument matrices is empty"))
    end

    if !all(Aₖ -> isCrispPCM(Aₖ), matrices)
        throw(ArgumentError("Aₖ is not a crisp PCM"))
    end
    
    l = length(matrices) # 人数
    m, n = size(matrices[1])

    if !all(Aₖ -> size(Aₖ) == (n, n), matrices)
        throw(ArgumentError("Some matrices have different size"))
    end

    ḋ = map(Aₖ -> solveCrispAHPLP(Aₖ).optimalValue, matrices)

    print(ḋ)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # wᵢᴸ ≥ ε, wᵢᵁ ≥ ε
        @variable(model, wᴸ[i=1:n] ≥ ε); @variable(model, wᵁ[i=1:n] ≥ ε)
        # vₖᵢᴸ ≥ ε, vₖᵢᵁ ≥ ε
        @variable(model, vᴸ[k=1:l,i=1:n] ≥ ε); @variable(model, vᵁ[k=1:l,i=1:n] ≥ ε)
        # s ≥ ε
        @variable(model, s[k=1:l] ≥ ε)

        for k = 1:l
            vₖᴸ = vᴸ[k,:]; vₖᵁ = vᵁ[k,:]

            Aₖ = matrices[k]

            # ∑(vₖᵢᵁ - vₖᵢᴸ) ≤ sₖḋₖ
            @constraint(model, sum(vₖᵁ) - sum(vₖᴸ) ≤ s[k]ḋ[k])

            for i = 1:n-1
                vₖᵢᴸ = vₖᴸ[i]; vₖᵢᵁ = vₖᵁ[i]
    
                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    vₖⱼᴸ = vₖᴸ[j]; vₖⱼᵁ = vₖᵁ[j]
                    
                    @constraint(model, vₖᵢᴸ ≤ aₖᵢⱼ * vₖⱼᵁ)
                    @constraint(model, aₖᵢⱼ * vₖⱼᴸ ≤ vₖᵢᵁ)
                end
            end

            for i = 1:n
                vₖᵢᴸ = vₖᴸ[i]; vₖᵢᵁ = vₖᵁ[i]

                # 正規性条件
                ∑vₖⱼᴸ = sum(map(j -> vₖᴸ[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑vₖⱼᴸ + vₖᵢᵁ ≤ 1)
                ∑vₖⱼᵁ = sum(map(j -> vₖᵁ[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑vₖⱼᵁ + vₖᵢᴸ ≥ 1)

                wᵢᴸ = wᴸ[i]; wᵢᵁ = wᵁ[i]
                @constraint(model, wᵢᴸ ≥ vₖᵢᴸ)
                @constraint(model, wᵢᵁ ≥ wᵢᴸ)
                @constraint(model, vₖᵢᵁ ≥ wᵢᵁ)
            end
        end

        for i = 1:n
            wᵢᴸ = wᴸ[i]; wᵢᵁ = wᵁ[i] 
            # kなしの正規性条件
            ∑wⱼᴸ = sum(map(j -> wᴸ[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼᴸ + wᵢᵁ ≤ 1)
            ∑wⱼᵁ = sum(map(j -> wᵁ[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼᵁ + wᵢᴸ ≥ 1)
        end

        # 目的関数 ∑(wᵢᵁ - wᵢᴸ)
        @objective(model, Max, sum(wᵁ) - sum(wᴸ))

        optimize!(model)

        optimalValue = sum(value.(wᵁ)) - sum(value.(wᴸ))

        wᴸ_value = value.(wᴸ)
        wᵁ_value = value.(wᵁ)
        # precision error 対応
        for i = 1:n
            if wᴸ_value[i] > wᵁ_value[i]
                wᴸ_value[i] = wᵁ_value[i]
            end
        end
        W_value = map(i -> (wᴸ_value[i])..(wᵁ_value[i]), 1:n)
        
        vᴸ_value = value.(vᴸ)
        vᵁ_value = value.(vᵁ)

        vᴸ_value ./= value.(s)
        vᵁ_value ./= value.(s)

        s_value = value.(s)

        # precision error 対応
        for k = 1:l, i = 1:n
            if vᴸ_value[k,i] > vᵁ_value[k,i]
                vᴸ_value[k,i] = vᵁ_value[k,i]
            end
        end
        v_value = map(
            k -> map(i -> (vᴸ_value[k,i])..(vᵁ_value[k,i]), 1:n),
            1:l)

        return (
            wᴸ=wᴸ_value, wᵁ=wᵁ_value,
            W=W_value,
            vᴸ=vᴸ_value, vᵁ=vᵁ_value,
            v=v_value,
            optimalValue=optimalValue,
            s=s_value
        )
    finally
        # エラー終了時にも変数などを消去する
        empty!(model)
    end
end
