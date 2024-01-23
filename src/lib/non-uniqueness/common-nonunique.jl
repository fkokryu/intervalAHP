using IntervalArithmetic
using JuMP
import HiGHS

include("../ttimes/optimal-value.jl")
include("../nearly-equal.jl")
include("../interval-ahp.jl")

LPResult_NonUniqueCommonGround = @NamedTuple{
    # 区間重みベクトル
    wᴸ_common_nonunique::Vector{T}, wᵁ_common_nonunique::Vector{T},
    W_common_nonunique::Vector{Interval{T}}, # ([wᵢᴸ_common_nonunique, wᵢᵁ_common_nonunique])
    ŵᴸ_common_nonunique::Matrix{T}, ŵᵁ_common_nonunique::Matrix{T},
    Ŵ_common_nonunique::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    optimalValue_common_nonunique::T
    } where {T <: Real}

function solveNonUniqueCommonGroundLP(matrices::Vector{Matrix{T}})::Union{LPResult_NonUniqueCommonGround{T}, Nothing} where {T <: Real}
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
    ḋ_common_nonunique = map(result -> result.optimalValue_center_1, w)
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
        # wᵢᴸ_common_nonunique ≥ ε, wᵢᵁ_common_nonunique ≥ ε
        @variable(model, wᴸ_common_nonunique[i=1:n] ≥ ε); @variable(model, wᵁ_common_nonunique[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_common_nonunique ≥ ε, ŵₖᵢᵁ_common_nonunique ≥ ε
        @variable(model, ŵᴸ_common_nonunique[k=1:l,i=1:n] ≥ ε); @variable(model, ŵᵁ_common_nonunique[k=1:l,i=1:n] ≥ ε)

        for k = 1:l
            ŵₖᴸ_common_nonunique = ŵᴸ_common_nonunique[k,:]; ŵₖᵁ_common_nonunique = ŵᵁ_common_nonunique[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_common_nonunique - ŵₖᵢᴸ_common_nonunique) ≤ ḋ_common_nonuniqueₖ
            @constraint(model, sum(ŵₖᵁ_common_nonunique) - sum(ŵₖᴸ_common_nonunique) ≥ (sum(ŵₖᴸ_common_nonunique) + sum(ŵₖᵁ_common_nonunique)) / 2 * t_L[k]*(ḋ_common_nonunique[k] + ε))
            @constraint(model, sum(ŵₖᵁ_common_nonunique) - sum(ŵₖᴸ_common_nonunique) ≤ (sum(ŵₖᴸ_common_nonunique) + sum(ŵₖᵁ_common_nonunique)) / 2 * t_U[k]*(ḋ_common_nonunique[k] + ε))

            for i = 1:n-1
                ŵₖᵢᴸ_common_nonunique = ŵₖᴸ_common_nonunique[i]; ŵₖᵢᵁ_common_nonunique = ŵₖᵁ_common_nonunique[i]
    
                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_common_nonunique = ŵₖᴸ_common_nonunique[j]; ŵₖⱼᵁ_common_nonunique = ŵₖᵁ_common_nonunique[j]
                    
                    @constraint(model, ŵₖᵢᴸ_common_nonunique ≤ aₖᵢⱼ * ŵₖⱼᵁ_common_nonunique)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_common_nonunique ≤ ŵₖᵢᵁ_common_nonunique)
                end
            end

            for i = 1:n
                ŵₖᵢᴸ_common_nonunique = ŵₖᴸ_common_nonunique[i]; ŵₖᵢᵁ_common_nonunique = ŵₖᵁ_common_nonunique[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_common_nonunique = sum(map(j -> ŵₖᴸ_common_nonunique[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_common_nonunique + ŵₖᵢᵁ_common_nonunique ≤ 1)
                ∑ŵₖⱼᵁ_common_nonunique = sum(map(j -> ŵₖᵁ_common_nonunique[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_common_nonunique + ŵₖᵢᴸ_common_nonunique ≥ 1)

                wᵢᴸ_common_nonunique = wᴸ_common_nonunique[i]; wᵢᵁ_common_nonunique = wᵁ_common_nonunique[i]
                @constraint(model, wᵢᴸ_common_nonunique ≥ ŵₖᵢᴸ_common_nonunique)
                @constraint(model, wᵢᵁ_common_nonunique ≥ wᵢᴸ_common_nonunique)
                @constraint(model, ŵₖᵢᵁ_common_nonunique ≥ wᵢᵁ_common_nonunique)
            end
        end

        for i = 1:n
            wᵢᴸ_common_nonunique = wᴸ_common_nonunique[i]; wᵢᵁ_common_nonunique = wᵁ_common_nonunique[i] 
            # kなしの正規性条件
            ∑wⱼᴸ_common_nonunique = sum(map(j -> wᴸ_common_nonunique[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼᴸ_common_nonunique + wᵢᵁ_common_nonunique ≤ 1)
            ∑wⱼᵁ_common_nonunique = sum(map(j -> wᵁ_common_nonunique[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼᵁ_common_nonunique + wᵢᴸ_common_nonunique ≥ 1)
        end

        # 目的関数 ∑(wᵢᵁ_common_nonunique - wᵢᴸ_common_nonunique)
        @objective(model, Max, sum(wᵁ_common_nonunique) - sum(wᴸ_common_nonunique))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_common_nonunique = sum(value.(wᵁ_common_nonunique)) - sum(value.(wᴸ_common_nonunique))

            wᴸ_common_nonunique_value = value.(wᴸ_common_nonunique)
            wᵁ_common_nonunique_value = value.(wᵁ_common_nonunique)
            # precision error 対応
            for i = 1:n
                if wᴸ_common_nonunique_value[i] > wᵁ_common_nonunique_value[i]
                    wᴸ_common_nonunique_value[i] = wᵁ_common_nonunique_value[i]
                end
            end
            W_common_nonunique_value = map(i -> (wᴸ_common_nonunique_value[i])..(wᵁ_common_nonunique_value[i]), 1:n)
            
            ŵᴸ_common_nonunique_value = value.(ŵᴸ_common_nonunique)
            ŵᵁ_common_nonunique_value = value.(ŵᵁ_common_nonunique)
            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_common_nonunique_value[k,i] > ŵᵁ_common_nonunique_value[k,i]
                    ŵᴸ_common_nonunique_value[k,i] = ŵᵁ_common_nonunique_value[k,i]
                end
            end
            Ŵ_common_nonunique_value = map(
                k -> map(i -> (ŵᴸ_common_nonunique_value[k,i])..(ŵᵁ_common_nonunique_value[k,i]), 1:n),
                1:l)

            return (
                wᴸ_common_nonunique=wᴸ_common_nonunique_value, wᵁ_common_nonunique=wᵁ_common_nonunique_value,
                W_common_nonunique=W_common_nonunique_value,
                ŵᴸ_common_nonunique=ŵᴸ_common_nonunique_value, ŵᵁ_common_nonunique=ŵᵁ_common_nonunique_value,
                Ŵ_common_nonunique=Ŵ_common_nonunique_value,
                optimalValue_common_nonunique=optimalValue_common_nonunique
            )
        else
            # 解が見つからなかった場合の処理
            println("The CommonGround_nonunique optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
