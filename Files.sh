#!/usr/bin/env bash

# 函数定义
rename_directories() {
  local base_dir=$1
  local dry_run=${2:-false}
  local log_file="Files.log"
  local first_level_pattern='3D_100.0G_*_Chain'
  local second_level_suffix='Slit'
  local third_level_old_suffix='1.0T_0.0Xi_8T20'
  local third_level_new_suffix='1.0T_0.0Xi_8T6-20'

  > "$log_file"
  # 使用find命令来深度限制在第一级目录
  /usr/bin/find "$base_dir" -maxdepth 1 -type d -name "$first_level_pattern" | while IFS= read -r first_level_dir; do
    /usr/bin/find "$first_level_dir" -maxdepth 1 -type d -name "*$second_level_suffix" | while IFS= read -r second_level_dir; do
      /usr/bin/find "$second_level_dir" -maxdepth 1 -type d -name "$third_level_old_suffix" | while IFS= read -r third_level_dir; do
        local new_third_level_dir="${third_level_dir/%$third_level_old_suffix/$third_level_new_suffix}"
        # 如果不是干运行，则实际执行重命名
        if [ "$dry_run" = false ]; then
          if mv "$third_level_dir" "$new_third_level_dir" 2>>"$log_file"; then
            echo "Renamed $third_level_dir to $new_third_level_dir" | tee -a "$log_file"
          else
            echo "Failed to rename $third_level_dir" | tee -a "$log_file"
          fi
        elif [ "$dry_run" = true ]; then
          echo "Would rename $third_level_dir to $new_third_level_dir" | tee -a "$log_file"
        fi
      done
    done
  done
}

# 第二个参数控制是否进行干运行测试，true为进行干运行测试，false或不提供为实际执行
rename_directories $(cd "../Simus" && pwd) false
