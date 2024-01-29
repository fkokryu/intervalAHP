using CSV
using DataFrames
using IntervalArithmetic
using JuMP
import HiGHS
import MathOptInterface as MOI  # これを追加

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../ttimes/optimal-value.jl")

function extract_relevant_data(input_file::String, trial_numbers::Vector{Int})
    # CSVファイルを読み込む
    df = CSV.read(input_file, DataFrame)

    # 列名のリスト
    column_names = [
        "Trial",
        "解の非唯一性考慮のCommonGround_wᴸ", 
        "解の非唯一性考慮のCommonGround_wᵁ", 
        "解の非唯一性考慮のCommonGround_wᴸ_1", 
        "解の非唯一性考慮のCommonGround_wᵁ_1", 
        "解の非唯一性考慮のCommonGround_wᴸ_2", 
        "解の非唯一性考慮のCommonGround_wᵁ_2", 
        "解の非唯一性考慮のCommonGround_wᴸ_3", 
        "解の非唯一性考慮のCommonGround_wᵁ_3"
    ]

    # 対象のトライアルのデータを抽出
    filtered_data = filter(row -> row.Trial in trial_numbers, df)

    # 必要な列だけを選択
    relevant_data = select(filtered_data, column_names)

    return relevant_data
end

function extract_relevant_data_2(input_file::String, trial_numbers::Vector{Int})
    # CSVファイルを読み込む
    df = CSV.read(input_file, DataFrame)

    # 列名のリスト
    column_names = [
        "Trial",
        "PCM",
        "式1の最適値", 
        "式10の最適値"
    ]

    # 対象のトライアルのデータを抽出
    filtered_data = filter(row -> row.Trial in trial_numbers, df)

    # 必要な列だけを選択
    relevant_data = select(filtered_data, column_names)

    return relevant_data
end

function extract_optimal_values(df::DataFrame)
    optimal_1 = Dict{Tuple{Int, Int}, Float64}()
    optimal_10 = Dict{Tuple{Int, Int}, Float64}()

    for row in eachrow(df)
        trial = row.Trial
        pcm = row.PCM
        opt_val_1 = row.式1の最適値
        opt_val_10 = row.式10の最適値

        optimal_1[(trial, pcm)] = opt_val_1
        optimal_10[(trial, pcm)] = opt_val_10
    end

    return optimal_1, optimal_10
end

function convert_to_vectors_and_matrices(df::DataFrame)
    # ベクトルと行列を初期化
    wᴸ_common = Vector{Float64}[]
    wᵁ_common = Vector{Float64}[]
    ŵᴸ_common = Matrix{Float64}[]
    ŵᵁ_common = Matrix{Float64}[]

    for row in eachrow(df)
        # 各列から文字列を抽出し、数値の配列に変換
        wᴸ = parse.(Float64, split(row.解の非唯一性考慮のCommonGround_wᴸ[2:end-1], ", "))
        wᵁ = parse.(Float64, split(row.解の非唯一性考慮のCommonGround_wᵁ[2:end-1], ", "))
        ŵᴸ_1 = parse.(Float64, split(row.解の非唯一性考慮のCommonGround_wᴸ_1[2:end-1], ", "))
        ŵᵁ_1 = parse.(Float64, split(row.解の非唯一性考慮のCommonGround_wᵁ_1[2:end-1], ", "))
        ŵᴸ_2 = parse.(Float64, split(row.解の非唯一性考慮のCommonGround_wᴸ_2[2:end-1], ", "))
        ŵᵁ_2 = parse.(Float64, split(row.解の非唯一性考慮のCommonGround_wᵁ_2[2:end-1], ", "))
        ŵᴸ_3 = parse.(Float64, split(row.解の非唯一性考慮のCommonGround_wᴸ_3[2:end-1], ", "))
        ŵᵁ_3 = parse.(Float64, split(row.解の非唯一性考慮のCommonGround_wᵁ_3[2:end-1], ", "))

        # ベクトルに追加
        push!(wᴸ_common, wᴸ)
        push!(wᵁ_common, wᵁ)

        # 行列に変換して追加
        push!(ŵᴸ_common, hcat(ŵᴸ_1, ŵᴸ_2, ŵᴸ_3))
        push!(ŵᵁ_common, hcat(ŵᵁ_1, ŵᵁ_2, ŵᵁ_3))
    end

    return wᴸ_common, wᵁ_common, ŵᴸ_common, ŵᵁ_common
end

function calculate_t(wᴸ_common::Vector{Vector{Float64}}, wᵁ_common::Vector{Vector{Float64}})
    n = length(wᴸ_common)
    t = Vector{Float64}(undef, n)

    for i in 1:n
        wᴸ = wᴸ_common[i]
        wᵁ = wᵁ_common[i]

        # t[i] の計算
        t[i] = sum((wᴸ + wᵁ) / 2)
    end

    return t
end

function calculate_t_for_each_dm(ŵᴸ_common::Vector{Matrix{Float64}}, ŵᵁ_common::Vector{Matrix{Float64}})
    n = length(ŵᴸ_common)
    t_values = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        ŵᴸ = ŵᴸ_common[i]
        ŵᵁ = ŵᵁ_common[i]

        # 各DMに対する t 値の計算
        t_dm = Vector{Float64}(undef, size(ŵᴸ, 1))
        for j in 1:3
            t_dm[j] = sum((ŵᴸ[:, j] + ŵᵁ[:, j]) / 2)
        end

        t_values[i] = t_dm
    end

    return t_values
end

# CSVファイルからデータを読み込む関数
function tcommon(input_file1::String, input_file2::String, output_file::String)
    # CSVファイルを読み込む
    df1 = CSV.read(input_file1, DataFrame)

    # "CommonGround_wᴸ" 列に数値が存在する行の "Trial" 列を選択
    selected_trials = df1[.!ismissing.(df1[:, :"解の非唯一性考慮のCommonGround_wᴸ"]) .&
                      .!(df1[:, :"解の非唯一性考慮のCommonGround_wᴸ"] .== "E"), :"Trial"]
    
    unique_trial_numbers = unique(selected_trials)

    relevant_data = extract_relevant_data(input_file1, unique_trial_numbers)

    wᴸ_common, wᵁ_common, ŵᴸ_common, ŵᵁ_common = convert_to_vectors_and_matrices(relevant_data)

    t = calculate_t(wᴸ_common, wᵁ_common)
    t_dm = calculate_t_for_each_dm(ŵᴸ_common, ŵᵁ_common)

    data = extract_relevant_data_2(input_file2, unique_trial_numbers)
    optimal_1, optimal_10 = extract_optimal_values(data)

    results_df = DataFrame(Trial = Int[], 統合した解の中心の総和 = Float64[], t_L = Float64[], t_U = Float64[], DM1 = Float64[], DM1の式1の最適値 = Float64[], DM1の積 = Float64[], DM1の式10の最適値 = Float64[], DM2 = Float64[], DM2の式1の最適値 = Float64[],
    DM2の積 = Float64[], DM2の式10の最適値 = Float64[], DM3 = Float64[], DM3の式1の最適値 = Float64[], DM3の積 = Float64[], DM3の式10の最適値 = Float64[])

    for i in 1:length(unique_trial_numbers)
        wᴸ = 1/t[i] * wᴸ_common[i]
        wᵁ = 1/t[i] * wᵁ_common[i]

        # `candidate` 配列の初期化
        candidate = zeros(Float64, 5)
        candidate2 = zeros(Float64, 5)

        for l = 1:5
            wᵢᴸ_check = wᴸ[l]
            ∑wⱼᵁ = sum(map(j -> wᵁ[j], filter(j -> l != j, 1:5)))
            candidate[l] = ∑wⱼᵁ + wᵢᴸ_check

            wᵢᵁ_check = wᵁ[l]
            ∑wⱼᴸ = sum(map(j -> wᴸ[j], filter(j -> l != j, 1:5)))
            candidate2[l] = ∑wⱼᴸ + wᵢᵁ_check
        end

        display(candidate)

        t_L = 1 / minimum(candidate) #式(10)の解に対するt^L        
        t_U = 1 / maximum(candidate2) #式(10)の解に対するt^U 

        # 結果をDataFrameに追加
        push!(results_df, (Trial = unique_trial_numbers[i], 統合した解の中心の総和 = t[i], t_L = t_L, t_U = t_U, DM1 = t_dm[i][1], DM1の式1の最適値 = optimal_1[(unique_trial_numbers[i],1)], DM1の積 = t_dm[i][1]*optimal_10[(unique_trial_numbers[i],1)], DM1の式10の最適値 = optimal_10[(unique_trial_numbers[i],1)], DM2 = t_dm[i][2], DM2の式1の最適値 = optimal_1[(unique_trial_numbers[i],2)],
    DM2の積 = t_dm[i][2]*optimal_10[(unique_trial_numbers[i],2)], DM2の式10の最適値 = optimal_10[(unique_trial_numbers[i],2)], DM3 = t_dm[i][3], DM3の式1の最適値 = optimal_1[(unique_trial_numbers[i],3)], DM3の積 = t_dm[i][3]*optimal_10[(unique_trial_numbers[i],3)], DM3の式10の最適値 = optimal_10[(unique_trial_numbers[i],3)]))
    end

    # 結果をCSVファイルに書き出し
    CSV.write(output_file, results_df)
end

