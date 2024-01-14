using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../interval-ahp.jl")

LPResult_PerfectIncorporation = @NamedTuple{
    # 区間重みベクトル
    wᴸ_perfect_entani::Vector{T}, wᵁ_perfect_entani::Vector{T},
    W_perfect_entani::Vector{Interval{T}}, # ([wᵢᴸ_perfect_entani, wᵢᵁ_perfect_entani])
    ŵᴸ_perfect_entani::Matrix{T}, ŵᵁ_perfect_entani::Matrix{T},
    Ŵ_perfect_entani::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    optimalValue_perfect_entani::T
    } where {T <: Real}

function solvePerfectIncorporationLP(matrices::Vector{Matrix{T}})::Union{LPResult_PerfectIncorporation{T}, Nothing} where {T <: Real}
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
        # wᵢᴸ_perfect_entani ≥ ε, wᵢᵁ_perfect_entani ≥ ε
        @variable(model, wᴸ_perfect_entani[i=1:n] ≥ ε); @variable(model, wᵁ_perfect_entani[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_perfect_entani ≥ ε, ŵₖᵢᵁ_perfect_entani ≥ ε
        @variable(model, ŵᴸ_perfect_entani[k=1:l,i=1:n] ≥ ε); @variable(model, ŵᵁ_perfect_entani[k=1:l,i=1:n] ≥ ε)

        for k = 1:l
            ŵₖᴸ_perfect_entani = ŵᴸ_perfect_entani[k,:]; ŵₖᵁ_perfect_entani = ŵᵁ_perfect_entani[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_perfect_entani - ŵₖᵢᴸ_perfect_entani) ≤ ḋₖ
            @constraint(model, sum(ŵₖᵁ_perfect_entani) - sum(ŵₖᴸ_perfect_entani) ≤ ḋ[k])

            for i = 1:n-1
                ŵₖᵢᴸ_perfect_entani = ŵₖᴸ_perfect_entani[i]; ŵₖᵢᵁ_perfect_entani = ŵₖᵁ_perfect_entani[i]
    
                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_perfect_entani = ŵₖᴸ_perfect_entani[j]; ŵₖⱼᵁ_perfect_entani = ŵₖᵁ_perfect_entani[j]
                    
                    @constraint(model, ŵₖᵢᴸ_perfect_entani ≤ aₖᵢⱼ * ŵₖⱼᵁ_perfect_entani)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_perfect_entani ≤ ŵₖᵢᵁ_perfect_entani)
                end
            end

            for i = 1:n
                ŵₖᵢᴸ_perfect_entani = ŵₖᴸ_perfect_entani[i]; ŵₖᵢᵁ_perfect_entani = ŵₖᵁ_perfect_entani[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_perfect_entani = sum(map(j -> ŵₖᴸ_perfect_entani[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_perfect_entani + ŵₖᵢᵁ_perfect_entani ≤ 1)
                ∑ŵₖⱼᵁ_perfect_entani = sum(map(j -> ŵₖᵁ_perfect_entani[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_perfect_entani + ŵₖᵢᴸ_perfect_entani ≥ 1)

                wᵢᴸ_perfect_entani = wᴸ_perfect_entani[i]; wᵢᵁ_perfect_entani = wᵁ_perfect_entani[i] 
                @constraint(model, ŵₖᵢᴸ_perfect_entani ≥ wᵢᴸ_perfect_entani)
                @constraint(model, ŵₖᵢᵁ_perfect_entani ≥ ŵₖᵢᴸ_perfect_entani)
                @constraint(model, wᵢᵁ_perfect_entani ≥ ŵₖᵢᵁ_perfect_entani)
            end
        end

        # 目的関数 ∑(wᵢᵁ_perfect_entani - wᵢᴸ_perfect_entani)
        @objective(model, Min, sum(wᵁ_perfect_entani) - sum(wᴸ_perfect_entani))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_perfect_entani = sum(value.(wᵁ_perfect_entani)) - sum(value.(wᴸ_perfect_entani))

            wᴸ_perfect_entani_value = value.(wᴸ_perfect_entani)
            wᵁ_perfect_entani_value = value.(wᵁ_perfect_entani)
            # precision error 対応
            for i = 1:n
                if wᴸ_perfect_entani_value[i] > wᵁ_perfect_entani_value[i]
                    wᴸ_perfect_entani_value[i] = wᵁ_perfect_entani_value[i]
                end
            end
            W_perfect_entani_value = map(i -> (wᴸ_perfect_entani_value[i])..(wᵁ_perfect_entani_value[i]), 1:n)
            
            ŵᴸ_perfect_entani_value = value.(ŵᴸ_perfect_entani)
            ŵᵁ_perfect_entani_value = value.(ŵᵁ_perfect_entani)
            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_perfect_entani_value[k,i] > ŵᵁ_perfect_entani_value[k,i]
                    ŵᴸ_perfect_entani_value[k,i] = ŵᵁ_perfect_entani_value[k,i]
                end
            end
            Ŵ_perfect_entani_value = map(
                k -> map(i -> (ŵᴸ_perfect_entani_value[k,i])..(ŵᵁ_perfect_entani_value[k,i]), 1:n),
                1:l)

            return (
                wᴸ_perfect_entani=wᴸ_perfect_entani_value, wᵁ_perfect_entani=wᵁ_perfect_entani_value,
                W_perfect_entani=W_perfect_entani_value,
                ŵᴸ_perfect_entani=ŵᴸ_perfect_entani_value, ŵᵁ_perfect_entani=ŵᵁ_perfect_entani_value,
                Ŵ_perfect_entani=Ŵ_perfect_entani_value,
                optimalValue_perfect_entani=optimalValue_perfect_entani
            )
        else
            # 解が見つからなかった場合の処理
            println("The PerfectIncorporation_entani optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
