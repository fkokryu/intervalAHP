using CSV
using DataFrames
using IntervalArithmetic
using JuMP
import HiGHS
import MathOptInterface as MOI
using Plots
using LinearAlgebra

# 既存のファイルをインクルード（必要に応じて）
# include("../crisp-pcm.jl")
# include("../nearly-equal.jl")
# include("../ttimes/optimal-value.jl")

function extract_interval_weights(input_file::String, trial_numbers::Vector{Int})
    # CSVファイルを読み込む
    df = CSV.read(input_file, DataFrame)
    
    # 対象のトライアルのデータを抽出
    filtered_data = filter(row -> row.Trial in trial_numbers, df)
    
    return filtered_data
end

function parse_weight_vector(weight_str::String)
    # 文字列が "E" の場合や欠損値の場合は空のベクトルを返す
    if ismissing(weight_str) || weight_str == "E" || weight_str == "Error"
        return Float64[]
    end
    
    # 空の配列 "[]" の場合も空のベクトルを返す
    if weight_str == "[]" || weight_str[2:end-1] == ""
        return Float64[]
    end
    
    # 文字列から括弧を取り除き、カンマで分割して数値に変換
    return parse.(Float64, split(weight_str[2:end-1], ", "))
end

# 修正版：動的に利用可能な列を検出して処理する関数
function convert_weights_to_matrices(df::DataFrame)
    n_rows = nrow(df)
    
    # 各手法・各DMの重みを格納する配列
    entani_wᴸ = Vector{Vector{Float64}}(undef, n_rows)
    entani_wᵁ = Vector{Vector{Float64}}(undef, n_rows)
    nonunique_wᴸ = Vector{Vector{Float64}}(undef, n_rows)
    nonunique_wᵁ = Vector{Vector{Float64}}(undef, n_rows)
    
    # 各DMの重みを格納する配列
    entani_wᴸ_dms = Vector{Union{Matrix{Float64}, Nothing}}(undef, n_rows)
    entani_wᵁ_dms = Vector{Union{Matrix{Float64}, Nothing}}(undef, n_rows)
    nonunique_wᴸ_dms = Vector{Union{Matrix{Float64}, Nothing}}(undef, n_rows)
    nonunique_wᵁ_dms = Vector{Union{Matrix{Float64}, Nothing}}(undef, n_rows)
    
    # 利用可能なDM列を動的に検出
    available_dm_indices = Int[]
    for i in 1:10  # 最大10個のDMまでチェック
        entani_col_L = Symbol("entani論文のCommonGround_wᴸ_$(i)")
        entani_col_U = Symbol("entani論文のCommonGround_wᵁ_$(i)")
        nonunique_col_L = Symbol("解の非唯一性考慮のCommonGround_wᴸ_$(i)")
        nonunique_col_U = Symbol("解の非唯一性考慮のCommonGround_wᵁ_$(i)")
        
        if hasproperty(df, entani_col_L) && hasproperty(df, entani_col_U) &&
           hasproperty(df, nonunique_col_L) && hasproperty(df, nonunique_col_U)
            push!(available_dm_indices, i)
        end
    end
    
    println("利用可能なDMインデックス: ", available_dm_indices)
    
    for i in 1:n_rows
        row = df[i, :]
        
        # entani論文の重み（デフォルトは空の配列）
        entani_wᴸ[i] = parse_weight_vector(string(row.entani論文のCommonGround_wᴸ))
        entani_wᵁ[i] = parse_weight_vector(string(row.entani論文のCommonGround_wᵁ))
        
        # 解の非唯一性考慮の重み（デフォルトは空の配列）
        nonunique_wᴸ[i] = parse_weight_vector(string(row.解の非唯一性考慮のCommonGround_wᴸ))
        nonunique_wᵁ[i] = parse_weight_vector(string(row.解の非唯一性考慮のCommonGround_wᵁ))
        
        # 初期値をNothingに設定
        entani_wᴸ_dms[i] = nothing
        entani_wᵁ_dms[i] = nothing
        nonunique_wᴸ_dms[i] = nothing
        nonunique_wᵁ_dms[i] = nothing
        
        # 利用可能なDMの重みを動的に取得
        entani_weights_L = Vector{Vector{Float64}}()
        entani_weights_U = Vector{Vector{Float64}}()
        nonunique_weights_L = Vector{Vector{Float64}}()
        nonunique_weights_U = Vector{Vector{Float64}}()
        
        for dm_idx in available_dm_indices
            # entani論文の各DMの重みを取得
            entani_col_L = Symbol("entani論文のCommonGround_wᴸ_$(dm_idx)")
            entani_col_U = Symbol("entani論文のCommonGround_wᵁ_$(dm_idx)")
            entani_wᴸ_dm = parse_weight_vector(string(row[entani_col_L]))
            entani_wᵁ_dm = parse_weight_vector(string(row[entani_col_U]))
            
            # 解の非唯一性の各DMの重みを取得
            nonunique_col_L = Symbol("解の非唯一性考慮のCommonGround_wᴸ_$(dm_idx)")
            nonunique_col_U = Symbol("解の非唯一性考慮のCommonGround_wᵁ_$(dm_idx)")
            nonunique_wᴸ_dm = parse_weight_vector(string(row[nonunique_col_L]))
            nonunique_wᵁ_dm = parse_weight_vector(string(row[nonunique_col_U]))
            
            # 有効なデータのみを追加
            if !isempty(entani_wᴸ_dm) && !isempty(entani_wᵁ_dm)
                push!(entani_weights_L, entani_wᴸ_dm)
                push!(entani_weights_U, entani_wᵁ_dm)
            end
            
            if !isempty(nonunique_wᴸ_dm) && !isempty(nonunique_wᵁ_dm)
                push!(nonunique_weights_L, nonunique_wᴸ_dm)
                push!(nonunique_weights_U, nonunique_wᵁ_dm)
            end
        end
        
        # 行列に変換して保存（有効なデータがある場合のみ）
        if !isempty(entani_weights_L) && !isempty(entani_weights_U)
            entani_wᴸ_dms[i] = hcat(entani_weights_L...)
            entani_wᵁ_dms[i] = hcat(entani_weights_U...)
        end
        
        if !isempty(nonunique_weights_L) && !isempty(nonunique_weights_U)
            nonunique_wᴸ_dms[i] = hcat(nonunique_weights_L...)
            nonunique_wᵁ_dms[i] = hcat(nonunique_weights_U...)
        end
    end
    
    return entani_wᴸ, entani_wᵁ, nonunique_wᴸ, nonunique_wᵁ, 
           entani_wᴸ_dms, entani_wᵁ_dms, nonunique_wᴸ_dms, nonunique_wᵁ_dms
end

# 区間重要度から区間PCMを計算する関数
function calculate_interval_pcm(wᴸ::Vector{Float64}, wᵁ::Vector{Float64})
    n = length(wᴸ)
    pcm_L = zeros(Float64, n, n)
    pcm_U = zeros(Float64, n, n)
    
    for i in 1:n
        for j in 1:n
            if i == j
                # 対角成分は[1,1]に設定
                pcm_L[i, j] = 1.0
                pcm_U[i, j] = 1.0
            else
                # 区間PCMの下限 = wiᴸ / wjᵁ
                pcm_L[i, j] = wᴸ[i] / wᵁ[j]
                # 区間PCMの上限 = wiᵁ / wjᴸ
                pcm_U[i, j] = wᵁ[i] / wᴸ[j]
            end
        end
    end
    
    return pcm_L, pcm_U
end

# PCM行列を文字列表現に変換する関数
function pcm_to_string(pcm::Matrix{Float64})
    rows, cols = size(pcm)
    result = "["
    
    for i in 1:rows
        result *= "["
        for j in 1:cols
            result *= string(round(pcm[i, j], digits=4))
            if j < cols
                result *= ", "
            end
        end
        result *= "]"
        if i < rows
            result *= ", "
        end
    end
    
    result *= "]"
    return result
end

function process_and_save_interval_pcms_common(input_file::String, output_file::String)
    # CSVファイルを読み込む
    df = CSV.read(input_file, DataFrame)
    
    # "entani論文のCommonGround_wᴸ" または "解の非唯一性考慮のCommonGround_wᴸ" 列に数値が存在する行を選択
    valid_rows = (.!ismissing.(df[:, :"entani論文のCommonGround_wᴸ"]) .&
                  .!(string.(df[:, :"entani論文のCommonGround_wᴸ"]) .== "E") .&
                  .!(string.(df[:, :"entani論文のCommonGround_wᴸ"]) .== "Error")) .| 
                 (.!ismissing.(df[:, :"解の非唯一性考慮のCommonGround_wᴸ"]) .&
                  .!(string.(df[:, :"解の非唯一性考慮のCommonGround_wᴸ"]) .== "E") .&
                  .!(string.(df[:, :"解の非唯一性考慮のCommonGround_wᴸ"]) .== "Error"))
                
    # Trial番号を抽出
    unique_trial_numbers = unique(df[valid_rows, :"Trial"])
    
    # 有効なデータを抽出
    relevant_data = extract_interval_weights(input_file, unique_trial_numbers)
    
    # 区間重要度を抽出
    entani_wᴸ, entani_wᵁ, nonunique_wᴸ, nonunique_wᵁ, 
    entani_wᴸ_dms, entani_wᵁ_dms, nonunique_wᴸ_dms, nonunique_wᵁ_dms = 
        convert_weights_to_matrices(relevant_data)
    
    # 利用可能なDMの数を動的に取得
    max_dms = 0
    for i in 1:length(entani_wᴸ_dms)
        if entani_wᴸ_dms[i] !== nothing
            max_dms = max(max_dms, size(entani_wᴸ_dms[i], 2))
        end
        if nonunique_wᴸ_dms[i] !== nothing
            max_dms = max(max_dms, size(nonunique_wᴸ_dms[i], 2))
        end
    end
    
    # 結果を格納するDataFrameを動的に構築
    columns = [:Trial, :Entani統合PCM_L, :Entani統合PCM_U, :非唯一性統合PCM_L, :非唯一性統合PCM_U]
    
    # 各DMのカラムを動的に追加
    for dm in 1:max_dms
        push!(columns, Symbol("Entani_DM$(dm)_PCM_L"))
        push!(columns, Symbol("Entani_DM$(dm)_PCM_U"))
        push!(columns, Symbol("非唯一性_DM$(dm)_PCM_L"))
        push!(columns, Symbol("非唯一性_DM$(dm)_PCM_U"))
    end
    
    results_df = DataFrame([name => (name == :Trial ? Int[] : String[]) for name in columns]...)
    
    # 各トライアルごとに区間PCMを計算
    for i in 1:length(unique_trial_numbers)
        trial = unique_trial_numbers[i]
        
        # 結果を格納する辞書
        result_row = Dict{Symbol, Any}()
        result_row[:Trial] = trial
        
        # 初期値を設定
        for col in columns[2:end]  # :Trial以外の全ての列
            result_row[col] = ""
        end
        
        # Entani論文の区間ウェイトが有効であることを確認
        if isassigned(entani_wᴸ, i) && !isempty(entani_wᴸ[i]) && isassigned(entani_wᵁ, i) && !isempty(entani_wᵁ[i])
            # Entani統合PCM
            entani_pcm_L, entani_pcm_U = calculate_interval_pcm(entani_wᴸ[i], entani_wᵁ[i])
            result_row[:Entani統合PCM_L] = pcm_to_string(entani_pcm_L)
            result_row[:Entani統合PCM_U] = pcm_to_string(entani_pcm_U)
            
            # 各DMのPCM (Entani)
            if isassigned(entani_wᴸ_dms, i) && entani_wᴸ_dms[i] !== nothing
                num_dms = size(entani_wᴸ_dms[i], 2)
                for dm in 1:num_dms
                    entani_dm_pcm_L, entani_dm_pcm_U = calculate_interval_pcm(
                        entani_wᴸ_dms[i][:, dm], entani_wᵁ_dms[i][:, dm])
                    
                    result_row[Symbol("Entani_DM$(dm)_PCM_L")] = pcm_to_string(entani_dm_pcm_L)
                    result_row[Symbol("Entani_DM$(dm)_PCM_U")] = pcm_to_string(entani_dm_pcm_U)
                end
            end
        end
        
        # 非唯一性区間ウェイトが有効であることを確認
        if isassigned(nonunique_wᴸ, i) && !isempty(nonunique_wᴸ[i]) && isassigned(nonunique_wᵁ, i) && !isempty(nonunique_wᵁ[i])
            # 非唯一性統合PCM
            nonunique_pcm_L, nonunique_pcm_U = calculate_interval_pcm(nonunique_wᴸ[i], nonunique_wᵁ[i])
            result_row[:非唯一性統合PCM_L] = pcm_to_string(nonunique_pcm_L)
            result_row[:非唯一性統合PCM_U] = pcm_to_string(nonunique_pcm_U)
            
            # 各DMのPCM (非唯一性)
            if isassigned(nonunique_wᴸ_dms, i) && nonunique_wᴸ_dms[i] !== nothing
                num_dms = size(nonunique_wᴸ_dms[i], 2)
                for dm in 1:num_dms
                    nonunique_dm_pcm_L, nonunique_dm_pcm_U = calculate_interval_pcm(
                        nonunique_wᴸ_dms[i][:, dm], nonunique_wᵁ_dms[i][:, dm])
                    
                    result_row[Symbol("非唯一性_DM$(dm)_PCM_L")] = pcm_to_string(nonunique_dm_pcm_L)
                    result_row[Symbol("非唯一性_DM$(dm)_PCM_U")] = pcm_to_string(nonunique_dm_pcm_U)
                end
            end
        end
        
        # 結果をDataFrameに追加
        push!(results_df, NamedTuple(result_row))
    end
    
    # 結果をCSVファイルに書き出し
    CSV.write(output_file, results_df)
    println("区間PCMの計算が完了し、$(output_file)に保存されました。")
end

"""
CSVファイルから区間PCMを読み込み、指定された条件に基づいて表示する関数
引数:
- csv_file: CSVファイルパス
- trial_number: 表示するトライアル番号
- method: 表示する手法名（"Entani", "非唯一性", "All"）
- dm_number: 表示する意思決定者番号（1, 2, 3, 0(統合PCM), -1(すべて)）
"""
function display_filtered_interval_pcm(csv_file::String, trial_number::Int; method::String="All", dm_number::Int=-1)
    # CSVファイルを読み込む
    df = CSV.read(csv_file, DataFrame)
    
    # 指定されたtrial番号の行を取得
    row = filter(row -> row.Trial == trial_number, df)
    
    if nrow(row) == 0
        println("Trial $(trial_number)のデータが見つかりませんでした。")
        return
    end
    
    # 対象のtrial行を取得
    trial_row = first(row)
    
    # フィルタリング条件に基づいてPCM行列のリストを作成
    pcm_pairs = []
    
    # 手法名に基づくフィルタリング
    methods_to_display = []
    if method == "All"
        methods_to_display = ["Entani", "非唯一性"]
    else
        methods_to_display = [method]
    end
    
    # 利用可能なDM番号を動的に検出
    available_dms = Int[]
    for col_name in names(df)
        if startswith(col_name, "Entani_DM") && endswith(col_name, "_PCM_L")
            dm_num = parse(Int, split(split(col_name, "_DM")[2], "_PCM")[1])
            if !(dm_num in available_dms)
                push!(available_dms, dm_num)
            end
        end
    end
    sort!(available_dms)
    
    # 意思決定者番号に基づくフィルタリング
    dm_numbers_to_display = []
    if dm_number == -1  # すべての意思決定者
        dm_numbers_to_display = [0; available_dms]  # 0は統合PCM
    else
        dm_numbers_to_display = [dm_number]
    end
    
    # フィルタリング条件に合致するPCMペアを選択
    for m in methods_to_display
        for dm in dm_numbers_to_display
            if dm == 0  # 統合PCM
                if m == "Entani" && hasproperty(trial_row, :Entani統合PCM_L)
                    push!(pcm_pairs, ("Entani統合PCM", trial_row.Entani統合PCM_L, trial_row.Entani統合PCM_U))
                elseif m == "非唯一性" && hasproperty(trial_row, :非唯一性統合PCM_L)
                    push!(pcm_pairs, ("非唯一性統合PCM", trial_row.非唯一性統合PCM_L, trial_row.非唯一性統合PCM_U))
                end
            else  # 個別DM
                if m == "Entani"
                    col_L = Symbol("Entani_DM$(dm)_PCM_L")
                    col_U = Symbol("Entani_DM$(dm)_PCM_U")
                    if hasproperty(trial_row, col_L) && hasproperty(trial_row, col_U)
                        push!(pcm_pairs, ("Entani_DM$(dm)_PCM", trial_row[col_L], trial_row[col_U]))
                    end
                elseif m == "非唯一性"
                    col_L = Symbol("非唯一性_DM$(dm)_PCM_L")
                    col_U = Symbol("非唯一性_DM$(dm)_PCM_U")
                    if hasproperty(trial_row, col_L) && hasproperty(trial_row, col_U)
                        push!(pcm_pairs, ("非唯一性_DM$(dm)_PCM", trial_row[col_L], trial_row[col_U]))
                    end
                end
            end
        end
    end
    
    # 選択されたPCMを表示
    if isempty(pcm_pairs)
        println("指定された条件に合致するPCMデータがありません。")
        return
    end
    
    for (name, pcm_L_str, pcm_U_str) in pcm_pairs
        if safe_string_check(pcm_L_str) && safe_string_check(pcm_U_str)
            println("\n### $(name) (Trial $(trial_number))")
            
            # 文字列からPCM行列を解析
            pcm_L = parse_matrix_string_improved(string(pcm_L_str))
            pcm_U = parse_matrix_string_improved(string(pcm_U_str))
            
            # 解析に成功した場合のみ表示
            if !isempty(pcm_L) && !isempty(pcm_U)
                # 区間PCM行列をLaTeX形式で表示
                display_interval_pcm_latex(pcm_L, pcm_U)
                
                # 数値表形式で表示
                display_interval_pcm_table(pcm_L, pcm_U)
            end
        end
    end
end

"""
文字列からPCM行列を解析する関数（改良版）
入れ子配列を適切に処理
"""
function parse_matrix_string_improved(matrix_str::String)
    if isempty(matrix_str)
        return Array{Float64}(undef, 0, 0)
    end
    
    try
        # 配列構造を評価
        raw_data = eval(Meta.parse(matrix_str))
        
        # 行数と列数を決定
        num_rows = length(raw_data)
        num_cols = length(raw_data[1])
        
        # 配列を初期化
        matrix = Array{Float64}(undef, num_rows, num_cols)
        
        # データを転送
        for i in 1:num_rows
            for j in 1:num_cols
                matrix[i, j] = raw_data[i][j]
            end
        end
        
        return matrix
    catch e
        println("行列の解析中にエラーが発生しました: $e")
        println("問題の文字列: $matrix_str")
        return Array{Float64}(undef, 0, 0)
    end
end

"""
区間PCM行列をLaTeX形式で表示する関数
"""
function display_interval_pcm_latex(pcm_L::Array{Float64,2}, pcm_U::Array{Float64,2})
    n = size(pcm_L, 1)
    
    # 行列の次元を確認
    if n == 0 || size(pcm_U, 1) == 0
        println("有効な行列データがありません。")
        return
    end
    
    # LaTeX形式で行列を表示
    println("\\begin{bmatrix}")
    
    for i in 1:n
        for j in 1:n
            print("[$(round(pcm_L[i,j], digits=4)), $(round(pcm_U[i,j], digits=4))]")
            if j < n
                print(" & ")
            end
        end
        
        if i < n
            println(" \\\\")
        else
            println("")
        end
    end
    
    println("\\end{bmatrix}")
    println()
end

"""
区間PCM行列を数値表形式で表示する関数
"""
function display_interval_pcm_table(pcm_L::Array{Float64,2}, pcm_U::Array{Float64,2})
    n = size(pcm_L, 1)
    
    # 行列の次元を確認
    if n == 0 || size(pcm_U, 1) == 0
        println("有効な行列データがありません。")
        return
    end
    
    # 表のヘッダー
    print("           ")
    for j in 1:n
        print("項目$(j)                ")
    end
    println()
    
    # 表の内容
    for i in 1:n
        print("項目$(i)    ")
        for j in 1:n
            print("[$(round(pcm_L[i,j], digits=4)), $(round(pcm_U[i,j], digits=4))]    ")
        end
        println()
    end
    println()
end

"""
特定の区間PCM行列を直接表示する関数
"""
function display_specific_pcm(L_str::String, U_str::String, name::String = "区間PCM行列")
    pcm_L = parse_matrix_string_improved(L_str)
    pcm_U = parse_matrix_string_improved(U_str)
    
    if !isempty(pcm_L) && !isempty(pcm_U)
        println("\n### $(name)")
        display_interval_pcm_latex(pcm_L, pcm_U)
        display_interval_pcm_table(pcm_L, pcm_U)
    else
        println("行列の解析に失敗しました。")
    end
end

# 区間重要度の中心を計算する関数
function calculate_weight_centers(wᴸ::Vector{Float64}, wᵁ::Vector{Float64})
    n = length(wᴸ)
    w_center = zeros(Float64, n)
    
    for i in 1:n
        w_center[i] = (wᴸ[i] + wᵁ[i]) / 2.0
    end
    
    return w_center
end

# 区間重要度を正規化する関数
function normalize_interval_weights(wᴸ::Vector{Float64}, wᵁ::Vector{Float64})
    # 中心値を計算
    w_center = calculate_weight_centers(wᴸ, wᵁ)
    # 中心値の総和
    center_sum = sum(w_center)
    
    # 正規化
    n = length(wᴸ)
    wᴸ_normalized = zeros(Float64, n)
    wᵁ_normalized = zeros(Float64, n)
    
    for i in 1:n
        wᴸ_normalized[i] = wᴸ[i] / center_sum
        wᵁ_normalized[i] = wᵁ[i] / center_sum
    end
    
    return wᴸ_normalized, wᵁ_normalized, w_center, center_sum
end

# t_L値とt_U値を計算する関数
function calculate_t_values(wᴸ::Vector{Float64}, wᵁ::Vector{Float64})
    n = length(wᴸ)
    candidate = zeros(Float64, n)
    candidate2 = zeros(Float64, n)
    
    for l in 1:n
        wᵢᴸ_check = wᴸ[l]
        ∑wⱼᵁ = sum(map(j -> wᵁ[j], filter(j -> l != j, 1:n)))
        candidate[l] = ∑wⱼᵁ + wᵢᴸ_check
        
        wᵢᵁ_check = wᵁ[l]
        ∑wⱼᴸ = sum(map(j -> wᴸ[j], filter(j -> l != j, 1:n)))
        candidate2[l] = ∑wⱼᴸ + wᵢᵁ_check
    end
    
    # 0や極端に小さい値でt_Lが無限大になることを防ぐための処理
    min_candidate = minimum(candidate)
    if min_candidate <= 1e-6  # 非常に小さい値または0
        println("警告: candidate配列に極端に小さい値があるためt_Lの計算が不安定です。")
        t_L = 1.0  # デフォルト値
    else
        t_L = 1 / min_candidate  # 式(10)の解に対するt^L
    end
    
    max_candidate2 = maximum(candidate2)
    if max_candidate2 <= 1e-6  # 非常に小さい値または0
        println("警告: candidate2配列に極端に小さい値があるためt_Uの計算が不安定です。")
        t_U = 1.0  # デフォルト値
    else
        t_U = 1 / max_candidate2  # 式(10)の解に対するt^U
    end

    println("t_L: ", t_L, ", t_U: ", t_U)
    
    return t_L, t_U
end

# 区間重要度にt値を掛ける関数
function apply_t_values(wᴸ::Vector{Float64}, wᵁ::Vector{Float64}, t_L::Float64, t_U::Float64)
    n = length(wᴸ)
    wᴸ_t_L = zeros(Float64, n)
    wᵁ_t_L = zeros(Float64, n)
    wᴸ_t_U = zeros(Float64, n)
    wᵁ_t_U = zeros(Float64, n)
    
    for i in 1:n
        wᴸ_t_L[i] = wᴸ[i] * t_L
        wᵁ_t_L[i] = wᵁ[i] * t_L
        wᴸ_t_U[i] = wᴸ[i] * t_U
        wᵁ_t_U[i] = wᵁ[i] * t_U
    end
    
    return wᴸ_t_L, wᵁ_t_L, wᴸ_t_U, wᵁ_t_U
end

# 拡張版のCSV保存関数
function process_and_save_interval_pcms_extended(input_file::String, output_file::String)
    # CSVファイルを読み込む
    df = CSV.read(input_file, DataFrame)
    
    # "entani論文のCommonGround_wᴸ" または "解の非唯一性考慮のCommonGround_wᴸ" 列に数値が存在する行を選択
    valid_rows = (.!ismissing.(df[:, :"entani論文のCommonGround_wᴸ"]) .&
                  .!(string.(df[:, :"entani論文のCommonGround_wᴸ"]) .== "E") .&
                  .!(string.(df[:, :"entani論文のCommonGround_wᴸ"]) .== "Error")) .| 
                 (.!ismissing.(df[:, :"解の非唯一性考慮のCommonGround_wᴸ"]) .&
                  .!(string.(df[:, :"解の非唯一性考慮のCommonGround_wᴸ"]) .== "E") .&
                  .!(string.(df[:, :"解の非唯一性考慮のCommonGround_wᴸ"]) .== "Error"))
                
    # Trial番号を抽出
    unique_trial_numbers = unique(df[valid_rows, :"Trial"])
    
    # 有効なデータを抽出
    relevant_data = extract_interval_weights(input_file, unique_trial_numbers)
    
    # 区間重要度を抽出
    entani_wᴸ, entani_wᵁ, nonunique_wᴸ, nonunique_wᵁ, 
    entani_wᴸ_dms, entani_wᵁ_dms, nonunique_wᴸ_dms, nonunique_wᵁ_dms = 
        convert_weights_to_matrices(relevant_data)
    
    # 利用可能なDMの数を動的に取得
    max_dms = 0
    for i in 1:length(entani_wᴸ_dms)
        if entani_wᴸ_dms[i] !== nothing
            max_dms = max(max_dms, size(entani_wᴸ_dms[i], 2))
        end
        if nonunique_wᴸ_dms[i] !== nothing
            max_dms = max(max_dms, size(nonunique_wᴸ_dms[i], 2))
        end
    end
    
    # 結果を格納するDataFrame - 拡張版を動的に構築
    columns = [:Trial, :Entani統合PCM_L, :Entani統合PCM_U, :非唯一性統合PCM_L, :非唯一性統合PCM_U,
               :非唯一性正規化PCM_L, :非唯一性正規化PCM_U, :非唯一性t_L_PCM_L, :非唯一性t_L_PCM_U,
               :非唯一性t_U_PCM_L, :非唯一性t_U_PCM_U]
    
    # 各DMのカラムを動的に追加
    for dm in 1:max_dms
        push!(columns, Symbol("Entani_DM$(dm)_PCM_L"))
        push!(columns, Symbol("Entani_DM$(dm)_PCM_U"))
        push!(columns, Symbol("非唯一性_DM$(dm)_PCM_L"))
        push!(columns, Symbol("非唯一性_DM$(dm)_PCM_U"))
    end
    
    results_df = DataFrame([name => (name == :Trial ? Int[] : String[]) for name in columns]...)
    
    # 各トライアルごとに区間PCMを計算
    for i in 1:length(unique_trial_numbers)
        trial = unique_trial_numbers[i]
        
        # 結果を格納する辞書
        result_row = Dict{Symbol, Any}()
        result_row[:Trial] = trial
        
        # 初期値を設定
        for col in columns[2:end]  # :Trial以外の全ての列
            result_row[col] = ""
        end
        
        # Entani論文の区間ウェイトが有効であることを確認
        if isassigned(entani_wᴸ, i) && !isempty(entani_wᴸ[i]) && isassigned(entani_wᵁ, i) && !isempty(entani_wᵁ[i])
            # Entani統合PCM
            entani_pcm_L, entani_pcm_U = calculate_interval_pcm(entani_wᴸ[i], entani_wᵁ[i])
            result_row[:Entani統合PCM_L] = pcm_to_string(entani_pcm_L)
            result_row[:Entani統合PCM_U] = pcm_to_string(entani_pcm_U)
            
            # 各DMのPCM (Entani)
            if isassigned(entani_wᴸ_dms, i) && entani_wᴸ_dms[i] !== nothing
                num_dms = size(entani_wᴸ_dms[i], 2)
                for dm in 1:num_dms
                    entani_dm_pcm_L, entani_dm_pcm_U = calculate_interval_pcm(
                        entani_wᴸ_dms[i][:, dm], entani_wᵁ_dms[i][:, dm])
                    
                    result_row[Symbol("Entani_DM$(dm)_PCM_L")] = pcm_to_string(entani_dm_pcm_L)
                    result_row[Symbol("Entani_DM$(dm)_PCM_U")] = pcm_to_string(entani_dm_pcm_U)
                end
            end
        end
        
        # 非唯一性区間ウェイトが有効であることを確認
        if isassigned(nonunique_wᴸ, i) && !isempty(nonunique_wᴸ[i]) && isassigned(nonunique_wᵁ, i) && !isempty(nonunique_wᵁ[i])
            # 非唯一性統合PCM（元の値）
            nonunique_pcm_L, nonunique_pcm_U = calculate_interval_pcm(nonunique_wᴸ[i], nonunique_wᵁ[i])
            result_row[:非唯一性統合PCM_L] = pcm_to_string(nonunique_pcm_L)
            result_row[:非唯一性統合PCM_U] = pcm_to_string(nonunique_pcm_U)
            
            # 区間重要度を正規化して新しいPCMを計算
            nonunique_wᴸ_normalized, nonunique_wᵁ_normalized, w_center, center_sum = 
                normalize_interval_weights(nonunique_wᴸ[i], nonunique_wᵁ[i])
            
            # 正規化した区間PCM
            nonunique_normalized_pcm_L, nonunique_normalized_pcm_U = 
                calculate_interval_pcm(nonunique_wᴸ_normalized, nonunique_wᵁ_normalized)
            result_row[:非唯一性正規化PCM_L] = pcm_to_string(nonunique_normalized_pcm_L)
            result_row[:非唯一性正規化PCM_U] = pcm_to_string(nonunique_normalized_pcm_U)
            
            # t_LとするためのUの計算
            t_L, t_U = calculate_t_values(nonunique_wᴸ_normalized, nonunique_wᵁ_normalized)
            
            # t_L値とt_U値を適用した区間重要度
            nonunique_wᴸ_t_L, nonunique_wᵁ_t_L, nonunique_wᴸ_t_U, nonunique_wᵁ_t_U = 
                apply_t_values(nonunique_wᴸ_normalized, nonunique_wᵁ_normalized, t_L, t_U)
            
            # t_L値を適用した区間PCM
            nonunique_t_L_pcm_L, nonunique_t_L_pcm_U = 
                calculate_interval_pcm(nonunique_wᴸ_t_L, nonunique_wᵁ_t_L)
            result_row[:非唯一性t_L_PCM_L] = pcm_to_string(nonunique_t_L_pcm_L)
            result_row[:非唯一性t_L_PCM_U] = pcm_to_string(nonunique_t_L_pcm_U)
            
            # t_U値を適用した区間PCM
            nonunique_t_U_pcm_L, nonunique_t_U_pcm_U = 
                calculate_interval_pcm(nonunique_wᴸ_t_U, nonunique_wᵁ_t_U)
            result_row[:非唯一性t_U_PCM_L] = pcm_to_string(nonunique_t_U_pcm_L)
            result_row[:非唯一性t_U_PCM_U] = pcm_to_string(nonunique_t_U_pcm_U)
            
            # 各DMのPCM (非唯一性)
            if isassigned(nonunique_wᴸ_dms, i) && nonunique_wᴸ_dms[i] !== nothing
                num_dms = size(nonunique_wᴸ_dms[i], 2)
                for dm in 1:num_dms
                    nonunique_dm_pcm_L, nonunique_dm_pcm_U = calculate_interval_pcm(
                        nonunique_wᴸ_dms[i][:, dm], nonunique_wᵁ_dms[i][:, dm])
                    
                    result_row[Symbol("非唯一性_DM$(dm)_PCM_L")] = pcm_to_string(nonunique_dm_pcm_L)
                    result_row[Symbol("非唯一性_DM$(dm)_PCM_U")] = pcm_to_string(nonunique_dm_pcm_U)
                end
            end
        end
        
        # 結果をDataFrameに追加
        push!(results_df, NamedTuple(result_row))
    end
    
    # 結果をCSVファイルに書き出し
    CSV.write(output_file, results_df)
    println("拡張区間PCMの計算が完了し、$(output_file)に保存されました。")
end

"""
CSVファイルから区間PCMを読み込み、指定された条件に基づいて表示する関数（拡張版）
引数:
- csv_file: CSVファイルパス
- trial_number: 表示するトライアル番号
- method: 表示する手法名（"Entani", "非唯一性", "非唯一性正規化", "非唯一性t_L", "非唯一性t_U", "All"）
- dm_number: 表示する意思決定者番号（1, 2, 3, 0(統合PCM), -1(すべて)）
"""
function display_filtered_interval_pcm_extended(csv_file::String, trial_number::Int; method::String="All", dm_number::Int=-1)
    # CSVファイルを読み込む
    df = CSV.read(csv_file, DataFrame)
    
    # 指定されたtrial番号の行を取得
    row = filter(row -> row.Trial == trial_number, df)
    
    if nrow(row) == 0
        println("Trial $(trial_number)のデータが見つかりませんでした。")
        return
    end
    
    # 対象のtrial行を取得
    trial_row = first(row)
    
    # フィルタリング条件に基づいてPCM行列のリストを作成
    pcm_pairs = []
    
    # 手法名に基づくフィルタリング
    methods_to_display = []
    if method == "All"
        methods_to_display = ["Entani", "非唯一性", "非唯一性正規化", "非唯一性t_L", "非唯一性t_U"]
    else
        methods_to_display = [method]
    end
    
    # 利用可能なDM番号を動的に検出
    available_dms = Int[]
    for col_name in names(df)
        if startswith(col_name, "Entani_DM") && endswith(col_name, "_PCM_L")
            dm_num = parse(Int, split(split(col_name, "_DM")[2], "_PCM")[1])
            if !(dm_num in available_dms)
                push!(available_dms, dm_num)
            end
        end
    end
    sort!(available_dms)
    
    # 意思決定者番号に基づくフィルタリング
    dm_numbers_to_display = []
    if dm_number == -1  # すべての意思決定者
        dm_numbers_to_display = [0; available_dms]  # 0は統合PCM
    else
        dm_numbers_to_display = [dm_number]
    end
    
    # フィルタリング条件に合致するPCMペアを選択
    for m in methods_to_display
        for dm in dm_numbers_to_display
            if dm == 0  # 統合PCM
                if m == "Entani" && hasproperty(trial_row, :Entani統合PCM_L)
                    push!(pcm_pairs, ("Entani統合PCM", trial_row.Entani統合PCM_L, trial_row.Entani統合PCM_U))
                elseif m == "非唯一性" && hasproperty(trial_row, :非唯一性統合PCM_L)
                    push!(pcm_pairs, ("非唯一性統合PCM", trial_row.非唯一性統合PCM_L, trial_row.非唯一性統合PCM_U))
                elseif m == "非唯一性正規化" && hasproperty(trial_row, :非唯一性正規化PCM_L)
                    push!(pcm_pairs, ("非唯一性正規化PCM", trial_row.非唯一性正規化PCM_L, trial_row.非唯一性正規化PCM_U))
                elseif m == "非唯一性t_L" && hasproperty(trial_row, :非唯一性t_L_PCM_L)
                    push!(pcm_pairs, ("非唯一性t_L_PCM", trial_row.非唯一性t_L_PCM_L, trial_row.非唯一性t_L_PCM_U))
                elseif m == "非唯一性t_U" && hasproperty(trial_row, :非唯一性t_U_PCM_L)
                    push!(pcm_pairs, ("非唯一性t_U_PCM", trial_row.非唯一性t_U_PCM_L, trial_row.非唯一性t_U_PCM_U))
                end
            else  # 個別DM
                if m == "Entani"
                    col_L = Symbol("Entani_DM$(dm)_PCM_L")
                    col_U = Symbol("Entani_DM$(dm)_PCM_U")
                    if hasproperty(trial_row, col_L) && hasproperty(trial_row, col_U)
                        push!(pcm_pairs, ("Entani_DM$(dm)_PCM", trial_row[col_L], trial_row[col_U]))
                    end
                elseif m == "非唯一性" || m == "非唯一性正規化" || m == "非唯一性t_L" || m == "非唯一性t_U"
                    col_L = Symbol("非唯一性_DM$(dm)_PCM_L")
                    col_U = Symbol("非唯一性_DM$(dm)_PCM_U")
                    if hasproperty(trial_row, col_L) && hasproperty(trial_row, col_U)
                        push!(pcm_pairs, ("非唯一性_DM$(dm)_PCM", trial_row[col_L], trial_row[col_U]))
                    end
                end
            end
        end
    end
    
    # 選択されたPCMを表示
    if isempty(pcm_pairs)
        println("指定された条件に合致するPCMデータがありません。")
        return
    end
    
    for (name, pcm_L_str, pcm_U_str) in pcm_pairs
        if !isempty(pcm_L_str) && !isempty(pcm_U_str)
            println("\n### $(name) (Trial $(trial_number))")
            
            # 文字列からPCM行列を解析
            pcm_L = parse_matrix_string_improved(pcm_L_str)
            pcm_U = parse_matrix_string_improved(pcm_U_str)
            
            # 解析に成功した場合のみ表示
            if !isempty(pcm_L) && !isempty(pcm_U)
                # 区間PCM行列をLaTeX形式で表示
                display_interval_pcm_latex(pcm_L, pcm_U)
                
                # 数値表形式で表示
                display_interval_pcm_table(pcm_L, pcm_U)
            end
        end
    end
end