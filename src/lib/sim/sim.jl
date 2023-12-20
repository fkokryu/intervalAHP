using Random
using LinearAlgebra
using Base.Threads

# 正規化された重要度ベクトルの生成関数
function generate_normalized_weight_vector(n)
    weights = rand(n)
    return weights / sum(weights)  # 重要度ベクトルの合計が1になるように正規化
end

# PCMの生成関数
function generate_pcm(weights)
    n = length(weights)
    pcm = ones(n, n)
    for i in 1:n
        for j in 1:n
            value = weights[i] / weights[j]
            if value < 1/9 || value > 9
                return nothing  # 条件を満たさない場合は再生成が必要
            end
            pcm[i, j] = value
        end
    end
    return pcm
end

# 対数変換と摂動の適用関数
function perturbate_pcm(pcm)
    log_pcm = log.(pcm)
    perturbed_log_pcm = log_pcm + randn(size(log_pcm)) * 0.01
    return exp.(perturbed_log_pcm)
end

# 整合性のチェック関数
function check_consistency(pcm)
    return maximum(abs.(pcm * inv(pcm) - I)) < 0.1
end

# 指定された数の整合性のあるPCMを生成する関数（マルチスレッド版）
function generate_consistent_pcms(n, desired_count, max_iterations)
    generated_pcms = Threads.Atomic{Int}(0)
    local_pcms = [Vector{Matrix{Float64}}() for _ in 1:nthreads()]

    @threads for iter in 1:max_iterations
        weights = generate_normalized_weight_vector(n)
        pcm = nothing
        while pcm === nothing
            pcm = generate_pcm(weights)
        end
        perturbed_pcm = perturbate_pcm(pcm)

        if check_consistency(perturbed_pcm)
            atomic_add!(generated_pcms, 1)
            push!(local_pcms[threadid()], perturbed_pcm)

            if generated_pcms[] >= desired_count
                break
            end
        end
    end

    return vcat(local_pcms...)
end
