#!/bin/bash

# 检查输入参数
if [ -z "$1" ]; then
    echo "使用方法: ./calc_epoch_loss.sh <log_file_path>"
    exit 1
fi

LOG_FILE="$1"

awk -F'loss=' '
/Tok Epoch/ && /loss=/ {
    # 正则匹配提取 Epoch 编号
    if (match($0, /Epoch ([0-9]+)\//, ep)) {
        epoch = ep[1]
        
        # 清理 loss 字段，去掉末尾的 ] 和换行符
        val = $2
        sub(/].*/, "", val)
        gsub(/[ \t\r\n]/, "", val)
        
        # 确保解析出来的是有效数值
        if (val ~ /^[0-9]+(\.[0-9]+)?$/) {
            sum[epoch] += val
            count[epoch]++
        }
    }
}
END {
    printf "\n%-10s | %-25s | %-10s\n", "Epoch", "Avg Loss", "Steps Count"
    print "--------------------------------------------------------"
    
    # 按照 Epoch 顺序打印结果
    for (i = 1; i <= 500; i++) {
        if (count[i] > 0) {
            avg = sum[i] / count[i]
            printf "Epoch %-4d | %-25.4f | %-10d\n", i, avg, count[i]
        }
    }
}
' "$LOG_FILE"
