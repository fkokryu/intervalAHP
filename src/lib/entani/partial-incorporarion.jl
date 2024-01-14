using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../interval-ahp.jl")

LPResult_PartialIncorporation = @NamedTuple{
    # 区間重みベクトル
    wᴸ_partial_entani::Vector{T}, wᵁ_partial_entani::Vector{T},
    W_partial_entani::Vector{Interval{T}}, # ([wᵢᴸ_partial_entani, wᵢᵁ_partial_entani])
    ŵᴸ_partial_entani::Matrix{T}, ŵᵁ_partial_entani::Matrix{T},
    Ŵ_partial_entani::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    w_partial_entani::Vector{Vector{T}},
    optimalValue_partial_entani::T
    } where {T <: Real}

function solvePartialIncorporationLP(
        matrices::Vector{Matrix{T}}
        )::Union{LPResult_PartialIncorporation{T}, Nothing} where {T <: Real}
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

    ḋ = map(Aₖ -> solveIntervalAHPLP(Aₖ).optimalValue, matrices)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # wᵢᴸ_partial_entani ≥ ε, wᵢᵁ_partial_entani ≥ ε
        @variable(model, wᴸ_partial_entani[i=1:n] ≥ ε)
        @variable(model, wᵁ_partial_entani[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_partial_entani ≥ ε, ŵₖᵢᵁ_partial_entani ≥ ε
        @variable(model, ŵᴸ_partial_entani[k=1:l,i=1:n] ≥ ε)
        @variable(model, ŵᵁ_partial_entani[k=1:l,i=1:n] ≥ ε)
        # wₖ_partial_entaniᵢ ≥ ε
        @variable(model, w_partial_entani[k=1:l,i=1:n] ≥ ε)

        for k = 1:l
            ŵₖᴸ_partial_entani = ŵᴸ_partial_entani[k,:]; ŵₖᵁ_partial_entani = ŵᵁ_partial_entani[k,:]
            wₖ_partial_entani = w_partial_entani[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_partial_entani - ŵₖᵢᴸ_partial_entani) ≤ ḋₖ
            @constraint(model, sum(ŵₖᵁ_partial_entani) - sum(ŵₖᴸ_partial_entani) ≤ ḋ[k])

            for i = 1:n-1
                ŵₖᵢᴸ_partial_entani = ŵₖᴸ_partial_entani[i]; ŵₖᵢᵁ_partial_entani = ŵₖᵁ_partial_entani[i]

                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_partial_entani = ŵₖᴸ_partial_entani[j]; ŵₖⱼᵁ_partial_entani = ŵₖᵁ_partial_entani[j]

                    @constraint(model, ŵₖᵢᴸ_partial_entani ≤ aₖᵢⱼ * ŵₖⱼᵁ_partial_entani)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_partial_entani ≤ ŵₖᵢᵁ_partial_entani)
                end
            end

            # 正規性条件
            @constraint(model, sum(wₖ_partial_entani) == 1)

            for i = 1:n
                ŵₖᵢᴸ_partial_entani = ŵₖᴸ_partial_entani[i]; ŵₖᵢᵁ_partial_entani = ŵₖᵁ_partial_entani[i]
                wᵢᴸ_partial_entani = wᴸ_partial_entani[i]; wᵢᵁ_partial_entani = wᵁ_partial_entani[i]
                wₖ_partial_entaniᵢ = wₖ_partial_entani[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_partial_entani = sum(map(j -> ŵₖᴸ_partial_entani[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_partial_entani + ŵₖᵢᵁ_partial_entani ≤ 1)
                ∑ŵₖⱼᵁ_partial_entani = sum(map(j -> ŵₖᵁ_partial_entani[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_partial_entani + ŵₖᵢᴸ_partial_entani ≥ 1)

                @constraint(model, wₖ_partial_entaniᵢ ≥ wᵢᴸ_partial_entani)
                @constraint(model, ŵₖᵢᵁ_partial_entani ≥ wₖ_partial_entaniᵢ)

                @constraint(model, wₖ_partial_entaniᵢ ≥ ŵₖᵢᴸ_partial_entani)
                @constraint(model, wᵢᵁ_partial_entani ≥ wₖ_partial_entaniᵢ)
            end
        end

        # 目的関数 ∑(wᵢᵁ_partial_entani - wᵢᴸ_partial_entani)
        @objective(model, Min, sum(wᵁ_partial_entani) - sum(wᴸ_partial_entani))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_partial_entani = sum(value.(wᵁ_partial_entani)) - sum(value.(wᴸ_partial_entani))

            wᴸ_partial_entani_value = value.(wᴸ_partial_entani); wᵁ_partial_entani_value = value.(wᵁ_partial_entani)
            # precision error 対応
            for i = 1:n
                if wᴸ_partial_entani_value[i] > wᵁ_partial_entani_value[i]
                    wᴸ_partial_entani_value[i] = wᵁ_partial_entani_value[i]
                end
            end
            W_partial_entani_value = map(i -> (wᴸ_partial_entani_value[i])..(wᵁ_partial_entani_value[i]), 1:n)

            ŵᴸ_partial_entani_value = value.(ŵᴸ_partial_entani); ŵᵁ_partial_entani_value = value.(ŵᵁ_partial_entani)
            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_partial_entani_value[k,i] > ŵᵁ_partial_entani_value[k,i]
                    ŵᴸ_partial_entani_value[k,i] = ŵᵁ_partial_entani_value[k,i]
                end
            end
            Ŵ_partial_entani_value = map(
                k -> map(i -> (ŵᴸ_partial_entani_value[k,i])..(ŵᵁ_partial_entani_value[k,i]), 1:n),
                1:l)

            w_partial_entani_value = map(k -> value.(w_partial_entani[k,:]), 1:l)

            return (
                wᴸ_partial_entani=wᴸ_partial_entani_value, wᵁ_partial_entani=wᵁ_partial_entani_value,
                W_partial_entani=W_partial_entani_value,
                ŵᴸ_partial_entani=ŵᴸ_partial_entani_value, ŵᵁ_partial_entani=ŵᵁ_partial_entani_value,
                Ŵ_partial_entani=Ŵ_partial_entani_value,
                w_partial_entani=w_partial_entani_value,
                optimalValue_partial_entani=optimalValue_partial_entani
            )
        else
            # 解が見つからなかった場合の処理
            println("The PartialIncorporation_entani optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
