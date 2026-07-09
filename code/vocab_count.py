#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
"""
统计语料中每个单词的出现次数，生成词汇表vocab.txt
"""

def get_counts(input_file, output_file, verbose=2, max_vocab=0, min_count=1):
    vocab_hash = {}
    token_count = 0
    unique_count = 0
    print("开始建立词汇表...", file=sys.stderr)
    if verbose > 1:
        print(f"已处理 {token_count} tokens.", end="", file=sys.stderr)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            for token in tokens:
                if token == "<unk>":
                    print("\n<unk> 语料中未发现vector found in corpus.", file=sys.stderr)
                    print("Please remove <unk>s from your corpus.", file=sys.stderr)
                    return 1
                if token in vocab_hash:
                    vocab_hash[token] += 1
                else:
                    vocab_hash[token] = 1
                    unique_count += 1
                token_count += 1
                if verbose > 1 and token_count % 100000 == 0:
                    print(f"\r已处理 {token_count} 个tokens.", end="", file=sys.stderr)
    if verbose > 1:
        print(f"\r已处理 {token_count} 个tokens.", file=sys.stderr)
        print(f"共发现 {unique_count} 个唯一单词.", file=sys.stderr)

    vocab_list = list(vocab_hash.items())

    if max_vocab > 0 and max_vocab < len(vocab_list):
        vocab_list.sort(key=lambda x: (-x[1], x[0]))
        vocab_list = vocab_list[:max_vocab]
    else:
        max_vocab = len(vocab_list)

    vocab_list.sort(key=lambda x: (-x[1], x[0]))

    filtered_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in vocab_list:
            if count < min_count:
                if verbose > 0:
                    print(f"最小计数为 {min_count}.", file=sys.stderr)
                break
            f.write(f"{word} {count}\n")
            filtered_count += 1

    if filtered_count == max_vocab and max_vocab < len(vocab_hash):
        if verbose > 0:
            print(f"Truncating vocabulary at size {max_vocab}.", file=sys.stderr)

    print(f"词汇表大小为 {filtered_count}.", file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(description='Simple tool to extract unigram counts')
    parser.add_argument('input_file', help='Input corpus file')
    parser.add_argument('output_file', help='Output vocabulary file')
    parser.add_argument('-verbose', type=int, default=2, choices=[0, 1, 2],
                        help='Set verbosity: 0, 1, or 2 (default)')
    parser.add_argument('-max-vocab', type=int, default=0,
                        help='Upper bound on vocabulary size')
    parser.add_argument('-min-count', type=int, default=1,
                        help='Lower limit such that words which occur fewer than '
                             '<int> times are discarded')

    args = parser.parse_args()

    return get_counts(args.input_file, args.output_file,
                      verbose=args.verbose,
                      max_vocab=args.max_vocab,
                      min_count=args.min_count)


if __name__ == '__main__':
    sys.exit(main())