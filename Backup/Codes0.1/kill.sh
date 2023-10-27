#!/bin/bash

# 检查参数个数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 start_job_id end_job_id"
    exit 1
fi

# 赋值参数
start_job_id=$1
end_job_id=$2

# 使用 seq 和 xargs 执行 bkill
echo "seq $start_job_id $end_job_id | xargs -I {} sh -c 'bkill {} || true'"
seq $start_job_id $end_job_id | xargs -I {} sh -c 'bkill {} || true'
