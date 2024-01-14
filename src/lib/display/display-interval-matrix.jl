using IntervalArithmetic

function intervalmatrixLaTeXString(A::Matrix{Interval{Float64}})::String
    m, n = size(A)

    # 各成分の LaTeX 表記を入れる
    A_str = fill("", (m, n))
    for i = 1:m, j = 1:n
        # 小数点第3位で四捨五入し、区間を表示
        lower = round(A[i, j].lo, digits=3)
        upper = round(A[i, j].hi, digits=3)
        A_str[i, j] = "\\left[ $lower, $upper \\right]"
    end

    str = "\\begin{pmatrix} "
    for i = 1:m
        for j = 1:n
            str *= A_str[i, j]
            if j != n
                str *= " & "  # 列の区切り
            elseif i != m
                str *= " \\\\"  # 行の区切り
            end
        end
    end
    str *= " \\end{pmatrix}"

    return str
end