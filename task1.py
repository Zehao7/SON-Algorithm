import sys
from pyspark import SparkContext
import time
import itertools
import math


def task1(case, support, input_path, output_path):
    start_time = time.time()

    # Read the file
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    # read rdd and filter header
    raw_rdd = sc.textFile(input_path).zipWithIndex().filter(lambda row_index: row_index[1] > 0).keys().map(lambda x: x.split(','))
    if case == '2':
        raw_rdd = raw_rdd.map(lambda x: [x[1], x[0]])
    
    baskets_rdd = raw_rdd.groupByKey().mapValues(set).map(lambda x: x[1])
    baskets_size = baskets_rdd.count()


    # SON Phase 1 Map and Reduce: Find local candidate itemsets
    def phase1(iterator):
        itemsets = list(iterator)
        return a_priori(itemsets, math.ceil(support * (len(itemsets) / baskets_size)))

    candidates_rdd = baskets_rdd.mapPartitions(phase1).distinct()
    candidates = candidates_rdd.collect()


    # SON Phase 2 Map and Reduce: Find true frequent itemsets
    def phase2(iterator):
        itemsets = list(iterator)
        counter = dict()
        counter_list = []
        for itemset in itemsets:
            for candidate in candidates:
                if candidate.issubset(itemset) :
                    counter[candidate] = counter.get(candidate, 0) + 1
        for candidate, count in counter.items():
            counter_list.append((candidate, count))
        return counter_list
    freq_itemsets_rdd = baskets_rdd.mapPartitions(phase2).reduceByKey(lambda a, b: a+b).filter(lambda counter: counter[1] >= support).map(lambda counter: counter[0])
    freq_itemsets = freq_itemsets_rdd.collect()


    # Writing the output file
    with open(output_path, 'w') as f:
        f.write("Candidates:\n")
        write_output_file(f, candidates)
        f.write("Frequent Itemsets:\n")
        write_output_file(f, freq_itemsets)
    f.close()


    end_time = time.time()
    total_time = end_time - start_time
    print("Duration: {0:.5f}".format(total_time))





def scan(candidates, support, baskets):
    freq_candidates = set()
    counter = dict()
    for basket in baskets:
        for candidate in candidates:
            if candidate.issubset(basket):
                counter[candidate] = counter.get(candidate, 0) + 1
                if counter[candidate] >= support:
                    freq_candidates.add(candidate)
    return freq_candidates


def generate(freq_itemsets, k):
    """
    Generate a set of candidate itemsets of size k.
    """
    candidates = set()
    for itemset1 in freq_itemsets:
        for itemset2 in freq_itemsets:
            if itemset1 != itemset2:
                new_itemset = itemset1.union(itemset2)
                if len(new_itemset) == k and new_itemset not in candidates:
                    candidates.add(new_itemset)
    return candidates


def prune(candidates, freq_itemsets, k):
    pruned_candidates = set()
    for candidate in candidates:
        valid = True
        for subset in itertools.combinations(candidate, k-1):
            if frozenset(subset) not in freq_itemsets:
                valid = False
                break
        if valid:
            pruned_candidates.add(candidate)
    return pruned_candidates


def a_priori(baskets, support):
    itemset = set(frozenset({item}) for basket in baskets for item in basket)
    freq_itemsets = set()
    k = 2

    cur_itemset = scan(itemset, support, baskets)
    while cur_itemset:
        for item in cur_itemset:
            freq_itemsets.add(item)
        candidates_set = prune(generate(cur_itemset, k), cur_itemset, k)
        cur_itemset = scan(candidates_set, support, baskets)
        k += 1
    return freq_itemsets


def join_list(itemsets):
    for i in range(len(itemsets)):
        itemsets[i] = str(tuple(itemsets[i]))
        if itemsets[i][-2] == ",":
            itemsets[i] = itemsets[i][:-2] + itemsets[i][-1]
    return ','.join(itemsets)


def write_output_file(file, itemsets):
    itemsets = list(map(lambda x: list(x), itemsets))
    for itemset in itemsets:
        itemset.sort()
    itemsets.sort(key = lambda x: (len(x), x))

    s = ""
    l, r = 0, 1
    for i in range(len(itemsets)-1):
        if len(itemsets[i]) < len(itemsets[i+1]):
            s += join_list(itemsets[l:i+1]) + '\n\n'
            l = r
        r += 1
    s += join_list(itemsets[l:r]) + '\n\n'
    file.write(s)





def main():
    case = sys.argv[1]
    support = int(sys.argv[2])
    input_path = sys.argv[3]
    output_path = sys.argv[4]

    # case = '1'
    # support = int('4')
    # input_path = "/Users/leoli/Desktop/small1.csv"
    # output_path = "/Users/leoli/Desktop/output4.txt"

    task1(case, support, input_path, output_path)



if __name__ == "__main__":
    main()


# spark-submit task1.py 1 4 "/Users/leoli/Desktop/small2.csv" "/Users/leoli/Desktop/output1.txt"
# spark-submit task1.py 2 9 "/Users/leoli/Desktop/small2.csv" "/Users/leoli/Desktop/output1.txt"

    
