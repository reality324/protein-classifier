#!/usr/bin/env python3
"""
检查 UniProt 三标签数据集大小
"""
import requests
import time
from datetime import datetime, timedelta


def estimate_dataset_size():
    """估算三标签蛋白质数据集大小"""

    print("="*70)
    print("UniProt 三标签蛋白质数据估算")
    print("="*70)

    base_url = "https://rest.uniprot.org/uniprotkb"

    # 测试不同查询的数据量
    queries = [
        {
            "name": "有 EC + 有定位 (无功能筛选)",
            "query": "ec:[1 TO 6] AND (comment(SCL) OR keyword(Location))",
        },
        {
            "name": "Swiss-Prot 有 EC + 有定位",
            "query": "reviewed:true AND ec:[1 TO 6] AND comment(SCL)",
        },
        {
            "name": "Swiss-Prot 三标签齐全",
            "query": "reviewed:true AND ec:[1 TO 6] AND comment(SCL) AND (comment(Function) OR keyword(Function))",
        },
        {
            "name": "人类蛋白质 三标签齐全",
            "query": "organism:9606 AND reviewed:true AND ec:[1 TO 6] AND comment(SCL)",
        },
        {
            "name": "酵母蛋白质 三标签齐全",
            "query": "organism:4932 AND reviewed:true AND ec:[1 TO 6] AND comment(SCL)",
        },
    ]

    results = []

    for q in queries:
        print(f"\n查询: {q['name']}")
        print(f"  Query: {q['query'][:60]}...")

        try:
            # 获取总数
            params = {
                "query": q["query"],
                "format": "list",
                "size": 1,
            }

            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            # 从 header 获取 total
            total = response.headers.get('x-total-results', '未知')

            print(f"  总数: {total}")

            # 估算文件大小
            if total.isdigit():
                total = int(total)
                # 每个蛋白质约 1KB (TSV 格式)
                size_mb = total * 1 / 1024
                # API 每秒最多请求 5 次
                time_seconds = total / 5 * 0.3  # 每次请求间隔 0.3 秒

                results.append({
                    "name": q["name"],
                    "count": total,
                    "size_mb": size_mb,
                    "time_minutes": time_seconds / 60,
                })

        except Exception as e:
            print(f"  错误: {e}")

    # 打印总结
    print("\n" + "="*70)
    print("数据量估算总结")
    print("="*70)

    print(f"\n{'查询名称':<45} {'数量':>12} {'文件大小':>12} {'下载时间':>12}")
    print("-"*85)

    for r in results:
        time_str = f"{r['time_minutes']:.1f} 分钟" if r['time_minutes'] < 60 else f"{r['time_minutes']/60:.1f} 小时"

        print(f"{r['name']:<45} {r['count']:>12,} {r['size_mb']:>11.1f} MB {time_str:>12}")

    # 推荐配置
    print("\n" + "="*70)
    print("推荐下载配置")
    print("="*70)

    print("""
方案1: 小规模测试
  - 数量: 5,000 条
  - 用途: 算法对比、代码测试
  - 预计时间: 15-20 分钟

方案2: 标准训练集
  - 数量: 20,000 条
  - 用途: 日常训练
  - 预计时间: 1-2 小时

方案3: 完整数据集
  - 数量: 50,000 条
  - 用途: 正式训练、论文实验
  - 预计时间: 3-5 小时

方案4: 全量数据
  - 数量: ~100,000 条 (Swiss-Prot 上限)
  - 用途: 追求最佳效果
  - 预计时间: 6-10 小时
""")


def check_api_rate_limit():
    """检查 API 速率限制"""
    print("\n" + "="*70)
    print("API 速率限制测试")
    print("="*70)

    base_url = "https://rest.uniprot.org/uniprotkb"

    # 测试连续请求
    times = []
    for i in range(5):
        start = time.time()
        response = requests.get(
            base_url,
            params={"query": "reviewed:true", "format": "list", "size": 1},
            timeout=30
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  请求 {i+1}: {elapsed:.2f}秒")

        if i < 4:
            time.sleep(0.5)

    avg_time = sum(times) / len(times)
    print(f"\n平均响应时间: {avg_time:.2f}秒")
    print(f"建议请求间隔: 0.3-0.5秒")

    # 估算批量下载时间
    print("\n批量下载时间估算:")
    for count in [1000, 10000, 50000, 100000]:
        # API 每次最多返回 500 条
        batches = (count + 499) // 500
        # 每次请求 + 间隔 = 0.5 秒
        total_seconds = batches * 0.5
        print(f"  {count:,} 条记录: {batches} 批请求, 约 {total_seconds/60:.1f} 分钟")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='估算数据集大小')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='快速估算 (跳过详细查询)')

    args = parser.parse_args()

    estimate_dataset_size()

    if not args.quick:
        check_api_rate_limit()


if __name__ == "__main__":
    main()
