using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../ttimes/optimal-value.jl")

LPResult_t_2_PartialIncorporation = @NamedTuple{
    # 区間重みベクトル
    wᴸ::Vector{T}, wᵁ::Vector{T},
    W::Vector{Interval{T}}, # ([wᵢᴸ, wᵢᵁ])
    ŵᴸ::Matrix{T}, ŵᵁ::Matrix{T},
    ŵ::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    Ŵ::Vector{Vector{T}},
    optimalValue::T,
    s::Vector{T}
    } where {T <: Real}

function solvetPartialIncorporationLP2(
        matrices::Vector{Matrix{T}}
        )::LPResult_t_2_PartialIncorporation{T} where {T <: Real}
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

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # wᵢᴸ ≥ ε, wᵢᵁ ≥ ε
        @variable(model, wᴸ[i=1:n] ≥ ε)
        @variable(model, wᵁ[i=1:n] ≥ ε)
        # ŵₖᵢᴸ ≥ ε, ŵₖᵢᵁ ≥ ε
        @variable(model, ŵᴸ[k=1:l,i=1:n] ≥ ε)
        @variable(model, ŵᵁ[k=1:l,i=1:n] ≥ ε)
        # Ŵₖᵢ ≥ ε
        @variable(model, Ŵ[k=1:l,i=1:n] ≥ ε)
        # s ≥ ε
        @variable(model, s[k=1:l] ≥ ε)

        for k = 1:l
            ŵₖᴸ = ŵᴸ[k,:]; ŵₖᵁ = ŵᵁ[k,:]
            Ŵₖ = Ŵ[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ - ŵₖᵢᴸ) ≤ sₖḋₖ
            @constraint(model, sum(ŵₖᵁ) - sum(ŵₖᴸ) == s[k]ḋ[k])

            for i = 1:n-1
                ŵₖᵢᴸ = ŵₖᴸ[i]; ŵₖᵢᵁ = ŵₖᵁ[i]

                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ = ŵₖᴸ[j]; ŵₖⱼᵁ = ŵₖᵁ[j]

                    @constraint(model, ŵₖᵢᴸ ≤ aₖᵢⱼ * ŵₖⱼᵁ)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ ≤ ŵₖᵢᵁ)
                end
            end

            @constraint(model, sum(wᵁ) + sum(wᴸ) == 2)

            # 正規性条件
            @constraint(model, sum(Ŵₖ) == 1)

            for i = 1:n
                ŵₖᵢᴸ = ŵₖᴸ[i]; ŵₖᵢᵁ = ŵₖᵁ[i]
                wᵢᴸ = wᴸ[i]; wᵢᵁ = wᵁ[i]
                Ŵₖᵢ = Ŵₖ[i]

                # 正規性条件
                ∑ŵₖⱼᴸ = sum(map(j -> ŵₖᴸ[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ + ŵₖᵢᵁ ≤ 1)
                ∑ŵₖⱼᵁ = sum(map(j -> ŵₖᵁ[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ + ŵₖᵢᴸ ≥ 1)

                @constraint(model, Ŵₖᵢ ≥ wᵢᴸ)
                @constraint(model, ŵₖᵢᵁ ≥ Ŵₖᵢ)

                @constraint(model, Ŵₖᵢ ≥ ŵₖᵢᴸ)
                @constraint(model, wᵢᵁ ≥ Ŵₖᵢ)
            end
            
            @constraint(model, s[k] == sum(ŵₖᴸ) + sum(ŵₖᵁ))
        end

        # 目的関数 ∑(wᵢᵁ - wᵢᴸ)
        @objective(model, Min, sum(wᵁ) - sum(wᴸ))

        optimize!(model)

        optimalValue = sum(value.(wᵁ)) - sum(value.(wᴸ))

        wᴸ_value = value.(wᴸ); wᵁ_value = value.(wᵁ)
        # precision error 対応
        for i = 1:n
            if wᴸ_value[i] > wᵁ_value[i]
                wᴸ_value[i] = wᵁ_value[i]
            end
        end
        W_value = map(i -> (wᴸ_value[i])..(wᵁ_value[i]), 1:n)

        ŵᴸ_value = value.(ŵᴸ); ŵᵁ_value = value.(ŵᵁ)
        ŵᴸ_value ./= value.(s)
        ŵᵁ_value ./= value.(s)

        # precision error 対応
        for k = 1:l, i = 1:n
            if ŵᴸ_value[k,i] > ŵᵁ_value[k,i]
                ŵᴸ_value[k,i] = ŵᵁ_value[k,i]
            end
        end

        ŵ_value = map(
            k -> map(i -> (ŵᴸ_value[k,i])..(ŵᵁ_value[k,i]), 1:n),
            1:l)

        Ŵ_value = map(k -> value.(Ŵ[k,:]), 1:l)

        s_value = value.(s)

        return (
            wᴸ=wᴸ_value, wᵁ=wᵁ_value,
            W=W_value,
            ŵᴸ=ŵᴸ_value, ŵᵁ=ŵᵁ_value,
            ŵ=ŵ_value,
            Ŵ=Ŵ_value,
            optimalValue=optimalValue,
            s=s_value
        )
    finally
        # エラー終了時にも変数などを消去する
        empty!(model)
    end
end
