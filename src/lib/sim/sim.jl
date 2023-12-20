using Random
using LinearAlgebra
using Base.Threads

# 正規化された重要度ベクトルの生成関数
function generate_normalized_weight_vector(n)
    weights = rand(n)
    return weights / sum(weights)
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

# 対数変換と摂動の適用関数
function perturbate_pcm(pcm, perturbation_strength)
    log_pcm = log.(pcm)
    perturbed_log_pcm = log_pcm + randn(size(log_pcm)) * perturbation_strength
    perturbed_pcm = exp.(perturbed_log_pcm)
    return enforce_pcm_constraints(perturbed_pcm)
end

# PCMの制約を適用する関数
function enforce_pcm_constraints(pcm)
    n = size(pcm, 1)
    for i in 1:n
        pcm[i, i] = 1.0  # 対角成分を1に設定
        for j in 1:n
            if i != j
                pcm[i, j] = clamp(pcm[i, j], 1/9, 9)  # 値を1/9から9の範囲に制限
                pcm[j, i] = 1 / pcm[i, j]  # 対称性を保持
            end
        end
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