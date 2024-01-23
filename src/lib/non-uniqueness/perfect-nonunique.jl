using IntervalArithmetic
using JuMP
import HiGHS

include("../ttimes/optimal-value.jl")
include("../nearly-equal.jl")
include("../interval-ahp.jl")

LPResult_NonUniquePerfectIncorporation = @NamedTuple{
    # 区間重みベクトル
    wᴸ_perfect_nonunique::Vector{T}, wᵁ_perfect_nonunique::Vector{T},
    W_perfect_nonunique::Vector{Interval{T}}, # ([wᵢᴸ_perfect_nonunique, wᵢᵁ_perfect_nonunique])
    ŵᴸ_perfect_nonunique::Matrix{T}, ŵᵁ_perfect_nonunique::Matrix{T},
    Ŵ_perfect_nonunique::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    optimalValue_perfect_nonunique::T
    } where {T <: Real}

function solveNonUniquePerfectIncorporationLP(matrices::Vector{Matrix{T}})::Union{LPResult_NonUniquePerfectIncorporation{T}, Nothing} where {T <: Real}
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
    ḋ_perfect_nonunique = map(result -> result.optimalValue_center_1, w)
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
        # wᵢᴸ_perfect_nonunique ≥ ε, wᵢᵁ_perfect_nonunique ≥ ε
        @variable(model, wᴸ_perfect_nonunique[i=1:n] ≥ ε); @variable(model, wᵁ_perfect_nonunique[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_perfect_nonunique ≥ ε, ŵₖᵢᵁ_perfect_nonunique ≥ ε
        @variable(model, ŵᴸ_perfect_nonunique[k=1:l,i=1:n] ≥ ε); @variable(model, ŵᵁ_perfect_nonunique[k=1:l,i=1:n] ≥ ε)

        for k = 1:l
            ŵₖᴸ_perfect_nonunique = ŵᴸ_perfect_nonunique[k,:]; ŵₖᵁ_perfect_nonunique = ŵᵁ_perfect_nonunique[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_perfect_nonunique - ŵₖᵢᴸ_perfect_nonunique) ≤ ḋ_perfect_nonuniqueₖ
            @constraint(model, sum(ŵₖᵁ_perfect_nonunique) - sum(ŵₖᴸ_perfect_nonunique) ≥ (sum(ŵₖᴸ_perfect_nonunique) + sum(ŵₖᵁ_perfect_nonunique)) / 2 * t_L[k]*(ḋ_perfect_nonunique[k] + ε))
            @constraint(model, sum(ŵₖᵁ_perfect_nonunique) - sum(ŵₖᴸ_perfect_nonunique) ≤ (sum(ŵₖᴸ_perfect_nonunique) + sum(ŵₖᵁ_perfect_nonunique)) / 2 * t_U[k]*(ḋ_perfect_nonunique[k] + ε))

            for i = 1:n-1
                ŵₖᵢᴸ_perfect_nonunique = ŵₖᴸ_perfect_nonunique[i]; ŵₖᵢᵁ_perfect_nonunique = ŵₖᵁ_perfect_nonunique[i]
    
                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_perfect_nonunique = ŵₖᴸ_perfect_nonunique[j]; ŵₖⱼᵁ_perfect_nonunique = ŵₖᵁ_perfect_nonunique[j]
                    
                    @constraint(model, ŵₖᵢᴸ_perfect_nonunique ≤ aₖᵢⱼ * ŵₖⱼᵁ_perfect_nonunique)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_perfect_nonunique ≤ ŵₖᵢᵁ_perfect_nonunique)
                end
            end

            for i = 1:n
                ŵₖᵢᴸ_perfect_nonunique = ŵₖᴸ_perfect_nonunique[i]; ŵₖᵢᵁ_perfect_nonunique = ŵₖᵁ_perfect_nonunique[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_perfect_nonunique = sum(map(j -> ŵₖᴸ_perfect_nonunique[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_perfect_nonunique + ŵₖᵢᵁ_perfect_nonunique ≤ 1)
                ∑ŵₖⱼᵁ_perfect_nonunique = sum(map(j -> ŵₖᵁ_perfect_nonunique[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_perfect_nonunique + ŵₖᵢᴸ_perfect_nonunique ≥ 1)

                wᵢᴸ_perfect_nonunique = wᴸ_perfect_nonunique[i]; wᵢᵁ_perfect_nonunique = wᵁ_perfect_nonunique[i] 
                @constraint(model, ŵₖᵢᴸ_perfect_nonunique ≥ wᵢᴸ_perfect_nonunique)
                @constraint(model, ŵₖᵢᵁ_perfect_nonunique ≥ ŵₖᵢᴸ_perfect_nonunique)
                @constraint(model, wᵢᵁ_perfect_nonunique ≥ ŵₖᵢᵁ_perfect_nonunique)
            end
        end

        # 目的関数 ∑(wᵢᵁ_perfect_nonunique - wᵢᴸ_perfect_nonunique)
        @objective(model, Min, sum(wᵁ_perfect_nonunique) - sum(wᴸ_perfect_nonunique))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_perfect_nonunique = sum(value.(wᵁ_perfect_nonunique)) - sum(value.(wᴸ_perfect_nonunique))

            wᴸ_perfect_nonunique_value = value.(wᴸ_perfect_nonunique)
            wᵁ_perfect_nonunique_value = value.(wᵁ_perfect_nonunique)
            # precision error 対応
            for i = 1:n
                if wᴸ_perfect_nonunique_value[i] > wᵁ_perfect_nonunique_value[i]
                    wᴸ_perfect_nonunique_value[i] = wᵁ_perfect_nonunique_value[i]
                end
            end
            W_perfect_nonunique_value = map(i -> (wᴸ_perfect_nonunique_value[i])..(wᵁ_perfect_nonunique_value[i]), 1:n)
            
            ŵᴸ_perfect_nonunique_value = value.(ŵᴸ_perfect_nonunique)
            ŵᵁ_perfect_nonunique_value = value.(ŵᵁ_perfect_nonunique)
            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_perfect_nonunique_value[k,i] > ŵᵁ_perfect_nonunique_value[k,i]
                    ŵᴸ_perfect_nonunique_value[k,i] = ŵᵁ_perfect_nonunique_value[k,i]
                end
            end
            Ŵ_perfect_nonunique_value = map(
                k -> map(i -> (ŵᴸ_perfect_nonunique_value[k,i])..(ŵᵁ_perfect_nonunique_value[k,i]), 1:n),
                1:l)

            return (
                wᴸ_perfect_nonunique=wᴸ_perfect_nonunique_value, wᵁ_perfect_nonunique=wᵁ_perfect_nonunique_value,
                W_perfect_nonunique=W_perfect_nonunique_value,
                ŵᴸ_perfect_nonunique=ŵᴸ_perfect_nonunique_value, ŵᵁ_perfect_nonunique=ŵᵁ_perfect_nonunique_value,
                Ŵ_perfect_nonunique=Ŵ_perfect_nonunique_value,
                optimalValue_perfect_nonunique=optimalValue_perfect_nonunique
            )
        else
            # 解が見つからなかった場合の処理
            println("The PerfectIncorporation_nonunique optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
