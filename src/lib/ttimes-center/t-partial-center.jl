using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../ttimes/optimal-value.jl")

LPResult_t_2_PartialIncorporation = @NamedTuple{
    # 区間重みベクトル
    wᴸ_tpartial_center_1::Vector{T}, wᵁ_tpartial_center_1::Vector{T},
    W_tpartial_center_1::Vector{Interval{T}}, # ([wᵢᴸ_tpartial_center_1, wᵢᵁ_tpartial_center_1])
    ŵᴸ_tpartial_center_1::Matrix{T}, ŵᵁ_tpartial_center_1::Matrix{T},
    ŵ_tpartial_center_1::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    Ŵ_tpartial_center_1::Vector{Vector{T}},
    optimalValue_tpartial_center_1::T,
    s_tpartial_center_1::T
    } where {T <: Real}

function solvetPartialIncorporationLP2(
        matrices::Vector{Matrix{T}}
        )::Union{LPResult_t_2_PartialIncorporation{T}, Nothing} where {T <: Real}
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

    ḋ_tpartial_center_1 = map(Aₖ -> solveCrispAHPLP(Aₖ).optimalValue_center_1, matrices)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # wᵢᴸ_tpartial_center_1 ≥ ε, wᵢᵁ_tpartial_center_1 ≥ ε
        @variable(model, wᴸ_tpartial_center_1[i=1:n] ≥ ε)
        @variable(model, wᵁ_tpartial_center_1[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_tpartial_center_1 ≥ ε, ŵₖᵢᵁ_tpartial_center_1 ≥ ε
        @variable(model, ŵᴸ_tpartial_center_1[k=1:l,i=1:n] ≥ ε)
        @variable(model, ŵᵁ_tpartial_center_1[k=1:l,i=1:n] ≥ ε)
        # Ŵ_tpartial_center_1ₖᵢ ≥ ε
        @variable(model, Ŵ_tpartial_center_1[k=1:l,i=1:n] ≥ ε)

        @variable(model, s_tpartial_center_1 ≥ ε)

        for k = 1:l
            ŵₖᴸ_tpartial_center_1 = ŵᴸ_tpartial_center_1[k,:]; ŵₖᵁ_tpartial_center_1 = ŵᵁ_tpartial_center_1[k,:]
            Ŵ_tpartial_center_1ₖ = Ŵ_tpartial_center_1[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_tpartial_center_1 - ŵₖᵢᴸ_tpartial_center_1) ≤ sₖḋ_tpartial_center_1ₖ
            @constraint(model, sum(ŵₖᵁ_tpartial_center_1) - sum(ŵₖᴸ_tpartial_center_1) ≤ ((sum(ŵₖᴸ_tpartial_center_1) + sum(ŵₖᵁ_tpartial_center_1)) / 2 * (ḋ_tpartial_center_1[k] + ε)))

            for i = 1:n-1
                ŵₖᵢᴸ_tpartial_center_1 = ŵₖᴸ_tpartial_center_1[i]; ŵₖᵢᵁ_tpartial_center_1 = ŵₖᵁ_tpartial_center_1[i]

                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_tpartial_center_1 = ŵₖᴸ_tpartial_center_1[j]; ŵₖⱼᵁ_tpartial_center_1 = ŵₖᵁ_tpartial_center_1[j]

                    @constraint(model, ŵₖᵢᴸ_tpartial_center_1 ≤ aₖᵢⱼ * ŵₖⱼᵁ_tpartial_center_1)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_tpartial_center_1 ≤ ŵₖᵢᵁ_tpartial_center_1)
                end
            end

            @constraint(model, sum(wᵁ_tpartial_center_1) + sum(wᴸ_tpartial_center_1) == 2)

            # 正規性条件
            @constraint(model, sum(Ŵ_tpartial_center_1ₖ) == s_tpartial_center_1)

            for i = 1:n
                ŵₖᵢᴸ_tpartial_center_1 = ŵₖᴸ_tpartial_center_1[i]; ŵₖᵢᵁ_tpartial_center_1 = ŵₖᵁ_tpartial_center_1[i]
                wᵢᴸ_tpartial_center_1 = wᴸ_tpartial_center_1[i]; wᵢᵁ_tpartial_center_1 = wᵁ_tpartial_center_1[i]
                Ŵ_tpartial_center_1ₖᵢ = Ŵ_tpartial_center_1ₖ[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_tpartial_center_1 = sum(map(j -> ŵₖᴸ_tpartial_center_1[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_tpartial_center_1 + ŵₖᵢᵁ_tpartial_center_1 ≤ s_tpartial_center_1)
                ∑ŵₖⱼᵁ_tpartial_center_1 = sum(map(j -> ŵₖᵁ_tpartial_center_1[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_tpartial_center_1 + ŵₖᵢᴸ_tpartial_center_1 ≥ s_tpartial_center_1)

                @constraint(model, Ŵ_tpartial_center_1ₖᵢ ≥ wᵢᴸ_tpartial_center_1)
                @constraint(model, ŵₖᵢᵁ_tpartial_center_1 ≥ Ŵ_tpartial_center_1ₖᵢ)

                @constraint(model, Ŵ_tpartial_center_1ₖᵢ ≥ ŵₖᵢᴸ_tpartial_center_1)
                @constraint(model, wᵢᵁ_tpartial_center_1 ≥ Ŵ_tpartial_center_1ₖᵢ)
            end
        end

        # 目的関数 ∑(wᵢᵁ_tpartial_center_1 - wᵢᴸ_tpartial_center_1)
        @objective(model, Min, sum(wᵁ_tpartial_center_1) - sum(wᴸ_tpartial_center_1))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_tpartial_center_1 = sum(value.(wᵁ_tpartial_center_1)) - sum(value.(wᴸ_tpartial_center_1))

            s_tpartial_center_1_value = value.(s_tpartial_center_1)
            optimalValue_tpartial_center_1 = optimalValue_tpartial_center_1 / s_tpartial_center_1_value

            wᴸ_tpartial_center_1_value = value.(wᴸ_tpartial_center_1) ./s_tpartial_center_1_value
            wᵁ_tpartial_center_1_value = value.(wᵁ_tpartial_center_1) ./s_tpartial_center_1_value
            # precision error 対応
            for i = 1:n
                if wᴸ_tpartial_center_1_value[i] > wᵁ_tpartial_center_1_value[i]
                    wᴸ_tpartial_center_1_value[i] = wᵁ_tpartial_center_1_value[i]
                end
            end
            W_tpartial_center_1_value = map(i -> (wᴸ_tpartial_center_1_value[i])..(wᵁ_tpartial_center_1_value[i]), 1:n)

            ŵᴸ_tpartial_center_1_value = value.(ŵᴸ_tpartial_center_1) ./s_tpartial_center_1_value
            ŵᵁ_tpartial_center_1_value = value.(ŵᵁ_tpartial_center_1) ./s_tpartial_center_1_value

            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_tpartial_center_1_value[k,i] > ŵᵁ_tpartial_center_1_value[k,i]
                    ŵᴸ_tpartial_center_1_value[k,i] = ŵᵁ_tpartial_center_1_value[k,i]
                end
            end

            ŵ_tpartial_center_1_value = map(
                k -> map(i -> (ŵᴸ_tpartial_center_1_value[k,i])..(ŵᵁ_tpartial_center_1_value[k,i]), 1:n),
                1:l)

            Ŵ_tpartial_center_1_value = map(k -> value.(Ŵ_tpartial_center_1[k,:]), 1:l)
            Ŵ_tpartial_center_1_value = Ŵ_tpartial_center_1_value ./s_tpartial_center_1_value

            return (
                wᴸ_tpartial_center_1=wᴸ_tpartial_center_1_value, wᵁ_tpartial_center_1=wᵁ_tpartial_center_1_value,
                W_tpartial_center_1=W_tpartial_center_1_value,
                ŵᴸ_tpartial_center_1=ŵᴸ_tpartial_center_1_value, ŵᵁ_tpartial_center_1=ŵᵁ_tpartial_center_1_value,
                ŵ_tpartial_center_1=ŵ_tpartial_center_1_value,
                Ŵ_tpartial_center_1=Ŵ_tpartial_center_1_value,
                optimalValue_tpartial_center_1=optimalValue_tpartial_center_1,
                s_tpartial_center_1=s_tpartial_center_1_value
            )
        else
            # 解が見つからなかった場合の処理
            println("The tPartialIncorporation2 optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
