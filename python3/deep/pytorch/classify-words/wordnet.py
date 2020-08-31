# -*- coding: utf-8 -*-
import sys
import argparse
from logzero import setup_logger
import sqlite3

sys.setrecursionlimit(10000)

class Wordnet():
    def __init__(self, config):
        self.config = config
        self.cache_collect_words = {}

    def create_hierarchy(self):
        hierarchy = {}  # key:上位語(String), value:下位語(List of String)

        cur = self.config.conn.execute("select synset1,synset2 from synlink where link='hypo'")  # 上位語-下位語の関係にあるものを抽出
        for row in cur:
            synset1 = row[0]
            synset2 = row[1]

            if synset1 not in hierarchy:
                hierarchy[synset1] = []

            hierarchy[synset1].append(synset2) 

        return hierarchy

    def collect_synsets(self, name):
        hierarchy = wordnet.create_hierarchy()
        cur = self.config.conn.execute("select synset from synset where name = '{}'".format(name))
        data = cur.fetchall()
        if len(data) == 0:
            self.config.logger.warning('"{}" is not included in synset table'.format(name))
            return

        output = set()
        for row in data:
            name = row[0]
            output |= self.__collect_synsets_recursive(name, hierarchy)
            print(output)

        query = 'select name from synset where synset in ({})'.format(','.join(f':{i}' for i in range(len(output))))
        cur = self.config.conn.execute(query, tuple(output))
        print([row[0] for row in cur.fetchall()])

    def __collect_synsets_recursive(self, synset, hierarchy):
        if synset not in hierarchy.keys():
            return set()

        if synset in self.cache_collect_words:
            return self.cache_collect_words[synset]

        output = set()
        hypos = set(hierarchy[synset])
        output |= hypos

        for hypo in hypos:
            output |= self.__collect_synsets_recursive(hypo, hierarchy)

        self.cache_collect_words[synset] = output
        return output

    def count_hypo(self):
        hierarchy = self.create_hierarchy()
        for synset in hierarchy.keys():
            output = self.__collect_synsets_recursive(synset, hierarchy)
            cur = self.config.conn.execute("select name from synset where synset = '{}'".format(synset))
            for row in cur:
                print('{}\t{}'.format(row[0], len(output)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('db', help='db file')
    parser.add_argument('--count_hypo', action='store_true', help='')
    parser.add_argument('--collect_synsets', default=None, help='')
    parser.add_argument('--loglevel', default='DEBUG')
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    args.conn = sqlite3.connect(args.db)

    wordnet = Wordnet(args)
    if args.count_hypo:
        wordnet.count_hypo()
        sys.exit()

    if args.collect_synsets is not None:
        wordnet.collect_synsets(args.collect_synsets)
        sys.exit()

