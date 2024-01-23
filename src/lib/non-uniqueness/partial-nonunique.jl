using IntervalArithmetic
using JuMP
import HiGHS

include("../ttimes/optimal-value.jl")
include("../nearly-equal.jl")
include("../interval-ahp.jl")

LPResult_NonUniquePartialIncorporation = @NamedTuple{
    # 区間重みベクトル
    wᴸ_partial_nonunique::Vector{T}, wᵁ_partial_nonunique::Vector{T},
    W_partial_nonunique::Vector{Interval{T}}, # ([wᵢᴸ_partial_nonunique, wᵢᵁ_partial_nonunique])
    ŵᴸ_partial_nonunique::Matrix{T}, ŵᵁ_partial_nonunique::Matrix{T},
    Ŵ_partial_nonunique::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    w_partial_nonunique::Vector{Vector{T}},
    optimalValue_partial_nonunique::T
    } where {T <: Real}

function solveNonUniquePartialIncorporationLP(
        matrices::Vector{Matrix{T}}
        )::Union{LPResult_NonUniquePartialIncorporation{T}, Nothing} where {T <: Real}
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

    w = map(Aₖ -> solveCrispAHPLP(Aₖ), matrices)
    ḋ_partial_nonunique = map(result -> result.optimalValue_center_1, w)
    wᴸ_individual = map(result -> result.wᴸ_center_1, w)
    wᵁ_individual = map(result -> result.wᵁ_center_1, w)

    # t_L と t_U の初期化
    t_L = fill(Inf, l)  # 最小値を見つけるために無限大で初期化
    t_U = fill(0.0, l)  # 最大値を見つけるために0で初期化

    for k = 1:l
        for i = 1:n
            ∑wⱼᵁ = sum(map(j -> wᵁ_individual[k][j], filter(j -> j != i, 1:n)))
            candidate = ∑wⱼᵁ + wᴸ_individual[k][i]

            ∑wⱼᴸ = sum(map(j -> wᴸ_individual[k][j], filter(j -> j != i, 1:n)))
            candidate2 = ∑wⱼᴸ + wᵁ_individual[k][i]

            # t_L と t_U の更新
            t_L[k] = min(t_L[k], 1 / candidate)
            t_U[k] = max(t_U[k], 1 / candidate2)
        end
    end

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # wᵢᴸ_partial_nonunique ≥ ε, wᵢᵁ_partial_nonunique ≥ ε
        @variable(model, wᴸ_partial_nonunique[i=1:n] ≥ ε)
        @variable(model, wᵁ_partial_nonunique[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_partial_nonunique ≥ ε, ŵₖᵢᵁ_partial_nonunique ≥ ε
        @variable(model, ŵᴸ_partial_nonunique[k=1:l,i=1:n] ≥ ε)
        @variable(model, ŵᵁ_partial_nonunique[k=1:l,i=1:n] ≥ ε)
        # wₖ_partial_nonuniqueᵢ ≥ ε
        @variable(model, w_partial_nonunique[k=1:l,i=1:n] ≥ ε)

        for k = 1:l
            ŵₖᴸ_partial_nonunique = ŵᴸ_partial_nonunique[k,:]; ŵₖᵁ_partial_nonunique = ŵᵁ_partial_nonunique[k,:]
            wₖ_partial_nonunique = w_partial_nonunique[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_partial_nonunique - ŵₖᵢᴸ_partial_nonunique) ≤ ḋ_partial_nonuniqueₖ
            @constraint(model, sum(ŵₖᵁ_partial_nonunique) - sum(ŵₖᴸ_partial_nonunique) ≥ (sum(ŵₖᴸ_partial_nonunique) + sum(ŵₖᵁ_partial_nonunique)) / 2 * t_L[k]*(ḋ_partial_nonunique[k] + ε))
            @constraint(model, sum(ŵₖᵁ_partial_nonunique) - sum(ŵₖᴸ_partial_nonunique) ≤ (sum(ŵₖᴸ_partial_nonunique) + sum(ŵₖᵁ_partial_nonunique)) / 2 * t_U[k]*(ḋ_partial_nonunique[k] + ε))

            for i = 1:n-1
                ŵₖᵢᴸ_partial_nonunique = ŵₖᴸ_partial_nonunique[i]; ŵₖᵢᵁ_partial_nonunique = ŵₖᵁ_partial_nonunique[i]

                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_partial_nonunique = ŵₖᴸ_partial_nonunique[j]; ŵₖⱼᵁ_partial_nonunique = ŵₖᵁ_partial_nonunique[j]

                    @constraint(model, ŵₖᵢᴸ_partial_nonunique ≤ aₖᵢⱼ * ŵₖⱼᵁ_partial_nonunique)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_partial_nonunique ≤ ŵₖᵢᵁ_partial_nonunique)
                end
            end

            # 正規性条件
            @constraint(model, sum(wₖ_partial_nonunique) == 1)

            for i = 1:n
                ŵₖᵢᴸ_partial_nonunique = ŵₖᴸ_partial_nonunique[i]; ŵₖᵢᵁ_partial_nonunique = ŵₖᵁ_partial_nonunique[i]
                wᵢᴸ_partial_nonunique = wᴸ_partial_nonunique[i]; wᵢᵁ_partial_nonunique = wᵁ_partial_nonunique[i]
                wₖ_partial_nonuniqueᵢ = wₖ_partial_nonunique[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_partial_nonunique = sum(map(j -> ŵₖᴸ_partial_nonunique[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_partial_nonunique + ŵₖᵢᵁ_partial_nonunique ≤ 1)
                ∑ŵₖⱼᵁ_partial_nonunique = sum(map(j -> ŵₖᵁ_partial_nonunique[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_partial_nonunique + ŵₖᵢᴸ_partial_nonunique ≥ 1)

                @constraint(model, wₖ_partial_nonuniqueᵢ ≥ wᵢᴸ_partial_nonunique)
                @constraint(model, ŵₖᵢᵁ_partial_nonunique ≥ wₖ_partial_nonuniqueᵢ)

                @constraint(model, wₖ_partial_nonuniqueᵢ ≥ ŵₖᵢᴸ_partial_nonunique)
                @constraint(model, wᵢᵁ_partial_nonunique ≥ wₖ_partial_nonuniqueᵢ)
            end
        end

        # 目的関数 ∑(wᵢᵁ_partial_nonunique - wᵢᴸ_partial_nonunique)
        @objective(model, Min, sum(wᵁ_partial_nonunique) - sum(wᴸ_partial_nonunique))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_partial_nonunique = sum(value.(wᵁ_partial_nonunique)) - sum(value.(wᴸ_partial_nonunique))

            wᴸ_partial_nonunique_value = value.(wᴸ_partial_nonunique); wᵁ_partial_nonunique_value = value.(wᵁ_partial_nonunique)
            # precision error 対応
            for i = 1:n
                if wᴸ_partial_nonunique_value[i] > wᵁ_partial_nonunique_value[i]
                    wᴸ_partial_nonunique_value[i] = wᵁ_partial_nonunique_value[i]
                end
            end
            W_partial_nonunique_value = map(i -> (wᴸ_partial_nonunique_value[i])..(wᵁ_partial_nonunique_value[i]), 1:n)

            ŵᴸ_partial_nonunique_value = value.(ŵᴸ_partial_nonunique); ŵᵁ_partial_nonunique_value = value.(ŵᵁ_partial_nonunique)
            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_partial_nonunique_value[k,i] > ŵᵁ_partial_nonunique_value[k,i]
                    ŵᴸ_partial_nonunique_value[k,i] = ŵᵁ_partial_nonunique_value[k,i]
                end
            end
            Ŵ_partial_nonunique_value = map(
                k -> map(i -> (ŵᴸ_partial_nonunique_value[k,i])..(ŵᵁ_partial_nonunique_value[k,i]), 1:n),
                1:l)

            w_partial_nonunique_value = map(k -> value.(w_partial_nonunique[k,:]), 1:l)

            return (
                wᴸ_partial_nonunique=wᴸ_partial_nonunique_value, wᵁ_partial_nonunique=wᵁ_partial_nonunique_value,
                W_partial_nonunique=W_partial_nonunique_value,
                ŵᴸ_partial_nonunique=ŵᴸ_partial_nonunique_value, ŵᵁ_partial_nonunique=ŵᵁ_partial_nonunique_value,
                Ŵ_partial_nonunique=Ŵ_partial_nonunique_value,
                w_partial_nonunique=w_partial_nonunique_value,
                optimalValue_partial_nonunique=optimalValue_partial_nonunique
            )
        else
            # 解が見つからなかった場合の処理
            println("The PartialIncorporation_nonunique optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
