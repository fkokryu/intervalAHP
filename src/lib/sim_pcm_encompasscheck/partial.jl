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
        "entani論文のPartialIncorporation_wᴸ", 
        "entani論文のPartialIncorporation_wᵁ", 
        "entani論文のPartialIncorporation_wᴸ_1", 
        "entani論文のPartialIncorporation_wᵁ_1", 
        "entani論文のPartialIncorporation_wᴸ_2", 
        "entani論文のPartialIncorporation_wᵁ_2", 
        "entani論文のPartialIncorporation_wᴸ_3", 
        "entani論文のPartialIncorporation_wᵁ_3"
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
    wᴸ_partial = Vector{Float64}[]
    wᵁ_partial = Vector{Float64}[]
    ŵᴸ_partial = Matrix{Float64}[]
    ŵᵁ_partial = Matrix{Float64}[]

    for row in eachrow(df)
        # 各列から文字列を抽出し、数値の配列に変換
        wᴸ = parse.(Float64, split(row.entani論文のPartialIncorporation_wᴸ[2:end-1], ", "))
        wᵁ = parse.(Float64, split(row.entani論文のPartialIncorporation_wᵁ[2:end-1], ", "))
        ŵᴸ_1 = parse.(Float64, split(row.entani論文のPartialIncorporation_wᴸ_1[2:end-1], ", "))
        ŵᵁ_1 = parse.(Float64, split(row.entani論文のPartialIncorporation_wᵁ_1[2:end-1], ", "))
        ŵᴸ_2 = parse.(Float64, split(row.entani論文のPartialIncorporation_wᴸ_2[2:end-1], ", "))
        ŵᵁ_2 = parse.(Float64, split(row.entani論文のPartialIncorporation_wᵁ_2[2:end-1], ", "))
        ŵᴸ_3 = parse.(Float64, split(row.entani論文のPartialIncorporation_wᴸ_3[2:end-1], ", "))
        ŵᵁ_3 = parse.(Float64, split(row.entani論文のPartialIncorporation_wᵁ_3[2:end-1], ", "))

        # ベクトルに追加
        push!(wᴸ_partial, wᴸ)
        push!(wᵁ_partial, wᵁ)

        # 行列に変換して追加
        push!(ŵᴸ_partial, hcat(ŵᴸ_1, ŵᴸ_2, ŵᴸ_3))
        push!(ŵᵁ_partial, hcat(ŵᵁ_1, ŵᵁ_2, ŵᵁ_3))
    end

    return wᴸ_partial, wᵁ_partial, ŵᴸ_partial, ŵᵁ_partial
end

function calculate_t(wᴸ_partial::Vector{Vector{Float64}}, wᵁ_partial::Vector{Vector{Float64}})
    n = length(wᴸ_partial)
    t = Vector{Float64}(undef, n)

    for i in 1:n
        wᴸ = wᴸ_partial[i]
        wᵁ = wᵁ_partial[i]

        # t[i] の計算
        t[i] = sum((wᴸ + wᵁ) / 2)
    end

    return t
end

function calculate_t_for_each_dm(ŵᴸ_partial::Vector{Matrix{Float64}}, ŵᵁ_partial::Vector{Matrix{Float64}})
    n = length(ŵᴸ_partial)
    t_values = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        ŵᴸ = ŵᴸ_partial[i]
        ŵᵁ = ŵᵁ_partial[i]

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
function partial(input_file1::String, input_file2::String, output_file::String)
    # CSVファイルを読み込む
    df1 = CSV.read(input_file1, DataFrame)

    # "PartialIncorporation_wᴸ" 列に数値が存在する行の "Trial" 列を選択
    selected_trials = df1[.!ismissing.(df1[:, :"entani論文のPartialIncorporation_wᴸ"]) .&
                      .!(df1[:, :"entani論文のPartialIncorporation_wᴸ"] .== "E"), :"Trial"]
    
    unique_trial_numbers = unique(selected_trials)

    relevant_data = extract_relevant_data(input_file1, unique_trial_numbers)

    wᴸ_partial, wᵁ_partial, ŵᴸ_partial, ŵᵁ_partial = convert_to_vectors_and_matrices(relevant_data)

    t = calculate_t(wᴸ_partial, wᵁ_partial)
    t_dm = calculate_t_for_each_dm(ŵᴸ_partial, ŵᵁ_partial)

    data = extract_relevant_data_2(input_file2, unique_trial_numbers)
    optimal_1, optimal_10 = extract_optimal_values(data)

    results_df = DataFrame(Trial = Int[], 統合した解の中心の総和 = Float64[], DM1 = Float64[], DM1の式1の最適値 = Float64[], DM1の積 = Float64[], DM1の式10の最適値 = Float64[], DM2 = Float64[], DM2の式1の最適値 = Float64[],
    DM2の積 = Float64[], DM2の式10の最適値 = Float64[], DM3 = Float64[], DM3の式1の最適値 = Float64[], DM3の積 = Float64[], DM3の式10の最適値 = Float64[])

    for i in 1:length(unique_trial_numbers)
        # 結果をDataFrameに追加

        push!(results_df, (Trial = unique_trial_numbers[i], 統合した解の中心の総和 = t[i], DM1 = t_dm[i][1], DM1の式1の最適値 = optimal_1[(unique_trial_numbers[i],1)], DM1の積 = t_dm[i][1]*optimal_10[(unique_trial_numbers[i],1)], DM1の式10の最適値 = optimal_10[(unique_trial_numbers[i],1)], DM2 = t_dm[i][2], DM2の式1の最適値 = optimal_1[(unique_trial_numbers[i],2)],
    DM2の積 = t_dm[i][2]*optimal_10[(unique_trial_numbers[i],2)], DM2の式10の最適値 = optimal_10[(unique_trial_numbers[i],2)], DM3 = t_dm[i][3], DM3の式1の最適値 = optimal_1[(unique_trial_numbers[i],3)], DM3の積 = t_dm[i][3]*optimal_10[(unique_trial_numbers[i],3)], DM3の式10の最適値 = optimal_10[(unique_trial_numbers[i],3)]))
    end

    # 結果をCSVファイルに書き出し
    CSV.write(output_file, results_df)
end

