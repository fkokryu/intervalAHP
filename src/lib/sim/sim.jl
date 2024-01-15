using Random
using LinearAlgebra
using Base.Threads
using Statistics
using Distributions

# 正規化された重要度ベクトルの生成関数（超平面上でのランダム点列の作り方による)
function generate_normalized_weight_vector(n)
    acceptable_ratio = false
    weights = Vector{Float64}(undef, n)

    while !acceptable_ratio
        random_num = [rand(Uniform(0, 1)) for _ in 1:n-1]
        sorted_random_num = sort(random_num)

        weights[1] = sorted_random_num[1]
        for i in 2:n-1
            weights[i] = sorted_random_num[i] - sorted_random_num[i-1]
        end
        weights[n] = 1 - sorted_random_num[n-1]

        max_weight = maximum(weights)
        min_weight = minimum(weights)

        if 8.5 <= max_weight / min_weight && max_weight / min_weight <= 9.5
            acceptable_ratio = true
        end
    end

    return weights
end

# Saatyのスケールに基づいて数値を離散化する関数
function discretize_saaty(a)
    if 2a <= -log(9) - log(8)
        return 1/9
    elseif 2a >= log(9) + log(8)
        return 9
    else
        for k in 2:8
            if -log(k + 1) - log(k) < 2a <= -log(k) - log(k - 1)
                return 1/k
            elseif log(k - 1) + log(k) <= 2a < log(k) + log(k + 1)
                return k
            end
        end
    end
    return 1  # Default return value if none of the conditions are met
end

# PCMの生成関数
function generate_pcm(weights)
    n = length(weights)
    pcm = ones(n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                # Calculate the ratio and discretize it
                pcm[i, j] = discretize_saaty(weights[i] / weights[j])
                pcm[j, i] = 1 / pcm[i, j]  # The matrix is reciprocal
            end
        end
    end
    return pcm
end

# PCMの摂動と離散化を適用する関数
function perturbate_and_discretize_pcm(pcm, perturbation_strength)
    n = size(pcm, 1)
    perturbed_pcm = copy(pcm)

    for i in 1:n
        for j in i+1:n  # 上三角行列の要素にのみ摂動を加える
            # 摂動を加える
            perturbed_value = log(pcm[i, j]) + rand(Uniform(-perturbation_strength, perturbation_strength))
            # 摂動後の値を離散化
            perturbed_pcm[i, j] = discretize_saaty(exp(perturbed_value))
            perturbed_pcm[j, i] = 1 / perturbed_pcm[i, j]  # 対称性を維持
        end
    end

    return perturbed_pcm
end

# 整合性のチェック関数
function check_consistency_saaty(pcm)
    n = size(pcm, 1)
    eigenvalues = eigvals(pcm)
    λ_max = maximum(real.(eigenvalues))
    CI = (λ_max - n) / (n - 1)
    return CI < 0.1
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

        perturbed_pcm = perturbate_and_discretize_pcm(original_pcm, perturbation_strength)
        if check_consistency_saaty(perturbed_pcm)
            if atomic_add!(generated_count, 1) <= desired_count
                push!(local_pcms[threadid()], perturbed_pcm)
            else
                break
            end
        end
    end

    return vcat(local_pcms...)
end