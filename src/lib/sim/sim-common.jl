using LinearAlgebra
using Base.Threads
using Statistics
using Distributions
using CSV
using DataFrames
using Random

# CSVファイルから区間重要度のデータを読み込む関数
function load_interval_weights_from_csv(file_path)
    df = CSV.read(file_path, DataFrame)
    interval_weights = []
    for i in 4:1000
        # 各区間重要度の左端と右端をペアにして追加（文字列から数値への変換を行う）
        row_weights = [(parse(Float64, df[i, 2*j]), parse(Float64, df[i, 2*j+1])) for j in 1:5]
        push!(interval_weights, row_weights)
    end
    return interval_weights
end

# ランダムに選ばれた行の区間重要度ベクトルを取得
function random_interval_weight_vector(interval_weights)
    return interval_weights[rand(1:length(interval_weights))]
end

# CSVファイルから区間重要度ベクトルを読み込む
file_path = "Simp.csv"  # CSVファイルのパス
interval_weights = load_interval_weights_from_csv(file_path)

# ランダムに選ばれた行の区間重要度ベクトルを取得
selected_interval_weight = random_interval_weight_vector(interval_weights)

# 特定の区間重要度からPCMを生成する関数
function generate_pcm_from_interval(selected_interval_weight)
    n = length(selected_interval_weight)
    pcm = ones(n, n)

    # 選ばれた区間重要度からランダムな値を選ぶ
    selected_values = [rand(Uniform(interval[1], interval[2])) for interval in selected_interval_weight]

    # 選ばれた値を使用してPCMを生成
    for i in 1:n
        for j in 1:n
            pcm[i, j] = selected_values[i] / selected_values[j]
        end
    end
    return pcm
end

# PCMに対数変換と摂動を適用する関数
function perturbate_pcm_from_interval(pcm, perturbation_strength)
    n = size(pcm, 1)
    perturbed_pcm = copy(pcm)

    for i in 1:n
        for j in i+1:n
            perturbed_value = log(pcm[i, j]) + randn() * perturbation_strength
            perturbed_pcm[i, j] = exp(perturbed_value)
            perturbed_pcm[j, i] = 1 / perturbed_pcm[i, j]
        end
    end

    return enforce_pcm_constraints_from_interval(perturbed_pcm)
end

# PCMの制約を適用する関数
function enforce_pcm_constraints_from_interval(pcm)
    n = size(pcm, 1)
    max_val = maximum(pcm)
    min_val = minimum(pcm)

    scale_factor = 1.0
    if max_val > 9
        scale_factor = min(scale_factor, 9 / max_val)
    end
    if min_val < 1/9
        scale_factor = min(scale_factor, min_val / (1/9))
    end

    for i in 1:n
        for j in 1:n
            if i != j
                pcm[i, j] = max(min(pcm[i, j] * scale_factor, 9), 1/9)
                pcm[j, i] = 1 / pcm[i, j]
            end
        end
    end
    return pcm
end

# PCMの整合性をチェックする関数
function check_consistency_from_interval(pcm)
    return maximum(abs.(pcm * inv(pcm) - I)) < 0.1
end

# 類似したPCMを生成する関数（マルチスレッド版）
function generate_similar_pcms_from_interval(perturbation_strength, desired_count)
    local_pcms = [Vector{Matrix{Float64}}() for _ in 1:nthreads()]
    selected_interval_weight = random_interval_weight_vector(interval_weights)
    original_pcm = generate_pcm_from_interval(selected_interval_weight)
    generated_count = Threads.Atomic{Int}(0)

    Threads.@threads for i in 1:desired_count * nthreads()
        if generated_count[] >= desired_count
            break
        end

        perturbed_pcm = perturbate_pcm_from_interval(original_pcm, perturbation_strength)
        if check_consistency_from_interval(perturbed_pcm)
            if atomic_add!(generated_count, 1) <= desired_count
                push!(local_pcms[threadid()], perturbed_pcm)
            else
                break
            end
        end
    end

    return vcat(local_pcms...)
end