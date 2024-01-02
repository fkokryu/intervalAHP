using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../ttimes/optimal-value.jl")

LPResult_one_PartialIncorporation = @NamedTuple{
    # 区間重みベクトル
    wᴸ::Vector{T}, wᵁ::Vector{T},
    W::Vector{Interval{T}}, # ([wᵢᴸ, wᵢᵁ])
    ŵᴸ::Matrix{T}, ŵᵁ::Matrix{T},
    Ŵ::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    w::Vector{Vector{T}},
    optimalValue::T
    } where {T <: Real}

function solveonePartialIncorporationLP(
        matrices::Vector{Matrix{T}}
        )::Union{LPResult_one_PartialIncorporation{T}, Nothing} where {T <: Real}
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
        # wₖᵢ ≥ ε
        @variable(model, w[k=1:l,i=1:n] ≥ ε)

        for k = 1:l
            ŵₖᴸ = ŵᴸ[k,:]; ŵₖᵁ = ŵᵁ[k,:]
            wₖ = w[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ - ŵₖᵢᴸ) ≤ ḋₖ
            @constraint(model, sum(ŵₖᵁ) - sum(ŵₖᴸ) ≤ ḋ[k])

            for i = 1:n-1
                ŵₖᵢᴸ = ŵₖᴸ[i]; ŵₖᵢᵁ = ŵₖᵁ[i]

                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ = ŵₖᴸ[j]; ŵₖⱼᵁ = ŵₖᵁ[j]

                    @constraint(model, ŵₖᵢᴸ ≤ aₖᵢⱼ * ŵₖⱼᵁ)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ ≤ ŵₖᵢᵁ)
                end
            end

            # 正規性条件
            @constraint(model, sum(wₖ) == 1)

            for i = 1:n
                ŵₖᵢᴸ = ŵₖᴸ[i]; ŵₖᵢᵁ = ŵₖᵁ[i]
                wᵢᴸ = wᴸ[i]; wᵢᵁ = wᵁ[i]
                wₖᵢ = wₖ[i]

                # 正規性条件
                ∑ŵₖⱼᴸ = sum(map(j -> ŵₖᴸ[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ + ŵₖᵢᵁ ≤ 1)
                ∑ŵₖⱼᵁ = sum(map(j -> ŵₖᵁ[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ + ŵₖᵢᴸ ≥ 1)

                @constraint(model, wₖᵢ ≥ wᵢᴸ)
                @constraint(model, ŵₖᵢᵁ ≥ wₖᵢ)

                @constraint(model, wₖᵢ ≥ ŵₖᵢᴸ)
                @constraint(model, wᵢᵁ ≥ wₖᵢ)
            end

            @constraint(model, sum(ŵₖᴸ) + sum(ŵₖᵁ) == 2)
        end

        @constraint(model, sum(wᵁ) + sum(wᴸ) == 2)

        # 目的関数 ∑(wᵢᵁ - wᵢᴸ)
        @objective(model, Min, sum(wᵁ) - sum(wᴸ))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
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
            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_value[k,i] > ŵᵁ_value[k,i]
                    ŵᴸ_value[k,i] = ŵᵁ_value[k,i]
                end
            end
            Ŵ_value = map(
                k -> map(i -> (ŵᴸ_value[k,i])..(ŵᵁ_value[k,i]), 1:n),
                1:l)

            w_value = map(k -> value.(w[k,:]), 1:l)

            return (
                wᴸ=wᴸ_value, wᵁ=wᵁ_value,
                W=W_value,
                ŵᴸ=ŵᴸ_value, ŵᵁ=ŵᵁ_value,
                Ŵ=Ŵ_value,
                w=w_value,
                optimalValue=optimalValue
            )
        else
            # 解が見つからなかった場合の処理
            println("The PartialIncorporation optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
