using Random
using LinearAlgebra
using Base.Threads
using Statistics
using Distributions

using Random

using Random

# 区間重要度ベクトルの生成関数
function generate_interval_weight_vector(n)
    # 初期化
    L = rand(n)  # 左端はランダムな正の値
    U = rand(n)  # 右端もランダムな正の値

    # 制約を満たすように調整
    for i in 1:n
        # 制約1: 各代替案の重要度の区間の右端と他の代替案の左端の和 >= 1
        while sum(U[1:n .!= i]) + L[i] < 1
            L[i] = rand()  # L[i] をランダムに再設定
        end

        # 制約2: 各代替案の重要度の区間の左端と他の代替案の右端の和 <= 1
        while sum(L[1:n .!= i]) + U[i] > 1
            U[i] = rand()  # U[i] をランダムに再設定
        end
    end

    return L, U
end



# PCMの生成関数
function generate_pcm(weights)
    n = length(weights)
    pcm = ones(n, n)
    for i in 1:n
        for j in 1:n
            pcm[i, j] = weights[i] / weights[j]
        end
    end
    return pcm
end

# 対数変換と摂動の適用関数（各要素に独立した摂動を加える）
function perturbate_pcm(pcm, perturbation_strength)
    n = size(pcm, 1)
    perturbed_pcm = copy(pcm)

    for i in 1:n
        for j in i+1:n  # 上三角行列の要素にのみ摂動を加える
            perturbed_value = log(pcm[i, j]) + randn() * perturbation_strength
            perturbed_pcm[i, j] = exp(perturbed_value)
            perturbed_pcm[j, i] = 1 / perturbed_pcm[i, j]  # 対称性を維持

            perturbed_value = 0
        end
    end

    return enforce_pcm_constraints(perturbed_pcm)
end

# PCMの制約を適用する関数
function enforce_pcm_constraints(pcm)
    n = size(pcm, 1)
    max_val = maximum(pcm)
    min_val = minimum(pcm)

    # スケーリング係数を計算
    scale_factor = 1.0
    if max_val > 9
        scale_factor = min(scale_factor, 9 / max_val)
    end
    if min_val < 1/9
        scale_factor = min(scale_factor, min_val / (1/9))
    end

    # 全要素にスケーリング係数を適用
    for i in 1:n
        for j in 1:n
            if i != j
                pcm[i, j] = max(min(pcm[i, j] * scale_factor, 9), 1/9)
                pcm[j, i] = 1 / pcm[i, j]
            end
        end
        pcm[i, i] = 1.0  # 対角成分を1に設定
    end
    return pcm
end

# 整合性のチェック関数
function check_consistency(pcm)
    return maximum(abs.(pcm * inv(pcm) - I)) < 0.1
end

# 指定された数の類似したPCMを生成する関数（マルチスレッド版）
function generate_similar_pcms(n, perturbation_strength, desired_count)
    local_pcms = [Vector{Matrix{Float64}}() for _ in 1:nthreads()]
    weights = generate_normalized_weight_vector(n)
    original_pcm = generate_pcm(weights)
    generated_count = Threads.Atomic{Int}(0)

    Threads.@threads for i in 1:desired_count * nthreads()
        if generated_count[] >= desired_count
            break
        end

        perturbed_pcm = perturbate_pcm(original_pcm, perturbation_strength)
        if check_consistency(perturbed_pcm)
            if atomic_add!(generated_count, 1) <= desired_count
                push!(local_pcms[threadid()], perturbed_pcm)
            else
                break
            end
        end
    end

    return vcat(local_pcms...)
end