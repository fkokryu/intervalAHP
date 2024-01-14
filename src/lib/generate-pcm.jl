using IntervalArithmetic

# 区間PCMを生成する関数
function create_interval_pcm(interval_values::Vector{Interval{Float64}})
    n = length(interval_values)
    pcm = Matrix{Interval{Float64}}(undef, n, n)

    for i in 1:n
        for j in 1:n
            if i == j
                pcm[i, j] = interval(1.0, 1.0)  # 対角要素は1
            elseif i < j
                pcm[i, j] = interval_values[j] / interval_values[i]  # 区間の比率
            else
                pcm[i, j] = 1 / pcm[j, i]  # 対称要素は逆数
            end
        end
    end

    return pcm
end
