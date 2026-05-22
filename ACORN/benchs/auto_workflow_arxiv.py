#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
自动化工作流：测试多种索引配置，自动调参并生成对比报告
"""

import os
import sys
import time
import subprocess
import json
import numpy as np
from datetime import datetime
from pathlib import Path


class AutoWorkflow:
    def __init__(self, dataset='arxiv_all', output_dir='results/auto_workflow', config_file=None):
        self.dataset = dataset
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 主日志文件（先初始化，以便后续可以使用 log 方法）
        self.main_log = os.path.join(self.output_dir, f'workflow_{self.timestamp}.log')
        self.summary_file = os.path.join(self.output_dir, f'summary_{self.timestamp}.txt')
        self.json_file = os.path.join(self.output_dir, f'results_{self.timestamp}.json')
        self.csv_file = os.path.join(self.output_dir, f'results_{self.timestamp}.csv')
        
        # 初始化日志文件
        with open(self.main_log, 'w') as f:
            f.write(f"Workflow started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 从配置文件加载或使用默认配置
        if config_file and os.path.exists(config_file):
            self.log(f"Loading config from {config_file}")
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.index_configs = config.get('index_configs', [])
                self.dataset = config.get('dataset', dataset)
                # 注意：如果配置文件指定了不同的输出目录，这里不会改变
        else:
            # 默认索引配置
            self.index_configs = [
                # IVF + PQ 配置（不同的聚类数和量化大小）
                {'name': 'IVF1024_PQ32', 'key': 'IVF1024,PQ32'},
                {'name': 'IVF2048_PQ32', 'key': 'IVF2048,PQ32'},
                {'name': 'IVF4096_PQ32', 'key': 'IVF4096,PQ32'},
                {'name': 'IVF4096_PQ64', 'key': 'IVF4096,PQ64'},
                {'name': 'IVF8192_PQ64', 'key': 'IVF8192,PQ64'},
                
                # Flat 作为基准（精确搜索）
                {'name': 'Flat', 'key': 'Flat'},
            ]
        
        # 存储所有结果
        self.results = {}
        
    def log(self, message):
        """写入日志"""
        timestamp_str = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp_str}] {message}"
        print(log_message)
        with open(self.main_log, 'a') as f:
            f.write(log_message + '\n')
    
    def run_benchmark(self, index_key, mode='autotune'):
        """运行单个基准测试"""
        script_path = os.path.join(os.path.dirname(__file__), 'bench_arxiv.py')
        
        if mode == 'autotune':
            cmd = ['python', script_path, self.dataset, index_key, 'autotune']
        else:
            # mode 应该是参数字符串，如 "nprobe=1,ht=2"
            cmd = ['python', script_path, self.dataset, index_key, mode]
        
        self.log(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=os.path.dirname(__file__)
            )
            return result.stdout, result.stderr, True
        except subprocess.CalledProcessError as e:
            self.log(f"Error running benchmark: {e}")
            return e.stdout, e.stderr, False
    
    def parse_autotune_results(self, output):
        """解析 autotune 输出，提取最优参数"""
        lines = output.split('\n')
        optimal_params = {}
        current_query_type = None
        
        for i, line in enumerate(lines):
            # 检测查询类型
            if 'Testing query type:' in line:
                query_type = line.split(':')[1].strip().lower()
                current_query_type = query_type
                optimal_params[query_type] = []
            
            # 解析最优参数（跳过第一个空参数）
            if current_query_type and 'nprobe=' in line or 'ht=' in line:
                parts = line.split()
                if len(parts) >= 3:
                    param = parts[0].strip()
                    recall = float(parts[1])
                    time_ms = float(parts[2])
                    if param and recall > 0:  # 过滤掉空参数和零召回
                        optimal_params[current_query_type].append({
                            'params': param,
                            'recall': recall,
                            'time': time_ms
                        })
        
        return optimal_params
    
    def parse_detailed_results(self, output):
        """解析详细测试结果"""
        lines = output.split('\n')
        results = {}
        current_query_type = None
        
        for line in lines:
            # 检测查询类型
            if 'Testing query type:' in line:
                query_type = line.split(':')[1].strip().lower()
                current_query_type = query_type
                results[query_type] = []
            
            # 解析性能指标
            if current_query_type and '\t' in line and 'R@1' not in line:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        param = parts[0]
                        r1 = float(parts[1])
                        r10 = float(parts[2])
                        r100 = float(parts[3])
                        time_ms = float(parts[4])
                        
                        results[current_query_type].append({
                            'params': param,
                            'R@1': r1,
                            'R@10': r10,
                            'R@100': r100,
                            'time_ms': time_ms
                        })
                    except (ValueError, IndexError):
                        continue
        
        return results
    
    def run_workflow(self):
        """运行完整的自动化工作流"""
        self.log("="*70)
        self.log("Starting Automated Benchmark Workflow")
        self.log(f"Dataset: {self.dataset}")
        self.log(f"Number of index configurations: {len(self.index_configs)}")
        self.log("="*70)
        
        for i, config in enumerate(self.index_configs, 1):
            index_name = config['name']
            index_key = config['key']
            
            self.log(f"\n{'='*70}")
            self.log(f"[{i}/{len(self.index_configs)}] Testing: {index_name} ({index_key})")
            self.log(f"{'='*70}")
            
            self.results[index_name] = {
                'index_key': index_key,
                'autotune': {},
                'detailed': {}
            }
            
            # 步骤 1: 运行 autotune
            self.log(f"Step 1: Running autotune for {index_name}...")
            start_time = time.time()
            output, stderr, success = self.run_benchmark(index_key, 'autotune')
            autotune_time = time.time() - start_time
            
            if not success:
                self.log(f"Autotune failed for {index_name}")
                continue
            
            self.log(f"Autotune completed in {autotune_time:.2f}s")
            
            # 解析 autotune 结果
            optimal_params = self.parse_autotune_results(output)
            self.results[index_name]['autotune'] = optimal_params
            
            # 显示找到的最优参数
            for query_type, params_list in optimal_params.items():
                self.log(f"  {query_type.upper()}: found {len(params_list)} optimal points")
                for p in params_list[:3]:  # 只显示前3个
                    self.log(f"    - {p['params']}: R@1={p['recall']:.4f}, time={p['time']:.3f}ms")
            
            # 步骤 2: 使用最优参数进行详细测试
            if optimal_params:
                self.log(f"Step 2: Running detailed tests with optimal parameters...")
                
                # 为每种查询类型选择最优参数（通常是召回率最高的那个）
                for query_type, params_list in optimal_params.items():
                    if params_list:
                        # 选择召回率最高的参数
                        best_param = max(params_list, key=lambda x: x['recall'])
                        param_str = best_param['params']
                        
                        self.log(f"  Testing {query_type.upper()} with {param_str}...")
                        output, stderr, success = self.run_benchmark(index_key, param_str)
                        
                        if success:
                            detailed = self.parse_detailed_results(output)
                            self.results[index_name]['detailed'][query_type] = detailed
                            self.log(f"    Detailed test completed for {query_type}")
        
        # 保存结果
        self.save_results()
        self.generate_summary()
        
        self.log("\n" + "="*70)
        self.log("Workflow completed successfully!")
        self.log(f"Summary saved to: {self.summary_file}")
        self.log(f"JSON results saved to: {self.json_file}")
        self.log(f"CSV results saved to: {self.csv_file}")
        self.log("="*70)
        
        # 显示快速摘要
        self.print_quick_summary()
    
    def save_results(self):
        """保存 JSON 格式的结果"""
        with open(self.json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.log(f"Results saved to {self.json_file}")
    
    def generate_summary(self):
        """生成汇总报告"""
        with open(self.summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Automated Benchmark Summary\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write("="*80 + "\n\n")
            
            # 为每种查询类型生成对比表
            query_types = ['equal', 'or', 'and']
            
            for qtype in query_types:
                f.write(f"\n{'='*80}\n")
                f.write(f"Query Type: {qtype.upper()}\n")
                f.write(f"{'='*80}\n\n")
                
                # 表头
                f.write(f"{'Index':<20} {'Best Params':<25} {'R@1':<8} {'R@10':<8} {'R@100':<8} {'Time(ms)':<10}\n")
                f.write("-"*80 + "\n")
                
                # 收集所有索引的结果
                for index_name, result in self.results.items():
                    autotune = result.get('autotune', {})
                    
                    if qtype in autotune and autotune[qtype]:
                        # 取召回率最高的参数
                        best = max(autotune[qtype], key=lambda x: x['recall'])
                        params = best['params']
                        recall = best['recall']
                        time_ms = best['time']
                        
                        # 如果有详细结果，使用详细结果
                        detailed = result.get('detailed', {}).get(qtype, {})
                        if qtype in detailed and detailed[qtype]:
                            det = detailed[qtype][0]
                            f.write(f"{index_name:<20} {params:<25} "
                                   f"{det['R@1']:<8.4f} {det['R@10']:<8.4f} "
                                   f"{det['R@100']:<8.4f} {det['time_ms']:<10.3f}\n")
                        else:
                            f.write(f"{index_name:<20} {params:<25} "
                                   f"{recall:<8.4f} {'N/A':<8} {'N/A':<8} {time_ms:<10.3f}\n")
                
                f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Recommendations:\n")
            f.write("="*80 + "\n\n")
            
            # 为每种查询类型推荐最佳配置
            for qtype in query_types:
                f.write(f"\n{qtype.upper()}:\n")
                
                best_index = None
                best_score = 0
                
                for index_name, result in self.results.items():
                    autotune = result.get('autotune', {})
                    if qtype in autotune and autotune[qtype]:
                        best = max(autotune[qtype], key=lambda x: x['recall'])
                        # 综合评分：召回率优先，时间次之
                        score = best['recall'] - (best['time'] / 1000)  # 简单评分
                        if score > best_score:
                            best_score = score
                            best_index = (index_name, best)
                
                if best_index:
                    name, params = best_index
                    f.write(f"  Best configuration: {name}\n")
                    f.write(f"  Optimal parameters: {params['params']}\n")
                    f.write(f"  R@1: {params['recall']:.4f}, Time: {params['time']:.3f}ms\n")
        
        self.log(f"Summary report saved to {self.summary_file}")
        
        # 生成 CSV 文件方便导入 Excel
        self.generate_csv()
    
    def generate_csv(self):
        """生成 CSV 格式的结果（便于在 Excel 中查看）"""
        import csv
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow(['Index', 'Query_Type', 'Best_Params', 'R@1', 'R@10', 'R@100', 'Time_ms'])
            
            # 写入数据
            query_types = ['equal', 'or', 'and']
            for index_name, result in self.results.items():
                autotune = result.get('autotune', {})
                
                for qtype in query_types:
                    if qtype in autotune and autotune[qtype]:
                        best = max(autotune[qtype], key=lambda x: x['recall'])
                        params = best['params']
                        recall = best['recall']
                        time_ms = best['time']
                        
                        # 如果有详细结果，使用详细结果
                        detailed = result.get('detailed', {}).get(qtype, {})
                        if qtype in detailed and detailed[qtype]:
                            det = detailed[qtype][0]
                            writer.writerow([
                                index_name, qtype, params,
                                f"{det['R@1']:.4f}", f"{det['R@10']:.4f}",
                                f"{det['R@100']:.4f}", f"{det['time_ms']:.3f}"
                            ])
                        else:
                            writer.writerow([
                                index_name, qtype, params,
                                f"{recall:.4f}", 'N/A', 'N/A', f"{time_ms:.3f}"
                            ])
        
        self.log(f"CSV results saved to {self.csv_file}")
    
    def print_quick_summary(self):
        """打印快速摘要到控制台"""
        self.log("\n" + "="*70)
        self.log("QUICK SUMMARY - Best Configurations")
        self.log("="*70)
        
        query_types = ['equal', 'or', 'and']
        
        for qtype in query_types:
            self.log(f"\n{qtype.upper()} Query:")
            
            best_index = None
            best_recall = 0
            
            for index_name, result in self.results.items():
                autotune = result.get('autotune', {})
                if qtype in autotune and autotune[qtype]:
                    best = max(autotune[qtype], key=lambda x: x['recall'])
                    if best['recall'] > best_recall:
                        best_recall = best['recall']
                        best_index = (index_name, best)
            
            if best_index:
                name, params = best_index
                self.log(f"  🏆 {name} - {params['params']}")
                self.log(f"     R@1: {params['recall']:.4f}, Time: {params['time']:.3f}ms")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='自动化索引测试工作流')
    parser.add_argument('--dataset', default='arxiv_all', help='数据集名称 (默认: arxiv_all)')
    parser.add_argument('--config', default=None, help='配置文件路径 (可选)')
    parser.add_argument('--output', default='results/auto_workflow', help='输出目录 (默认: results/auto_workflow)')
    
    args = parser.parse_args()
    
    workflow = AutoWorkflow(
        dataset=args.dataset,
        output_dir=args.output,
        config_file=args.config
    )
    workflow.run_workflow()


if __name__ == '__main__':
    main()

