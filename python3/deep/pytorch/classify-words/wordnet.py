# -*- coding: utf-8 -*-
import sys
import argparse
from logzero import setup_logger
import sqlite3

class WordnetModel():
    def __init__(self, config):
        self.config = config
        self.conn = self.config.conn

    def select_word_by_synset(self, synset):
        query = (
            "select synset.name, word.lemma " 
            "from (synset inner join sense on synset.synset = sense.synset) "
              "inner join word on sense.wordid = word.wordid "
            "where synset.synset = '{}' and word.lang = 'jpn' "
        ).format(synset)
        return self.conn.execute(query)

    def select_synset_by_word(self, word):
        query = (
            "select synset.synset, synset.name " 
            "from (synset inner join sense on synset.synset = sense.synset) "
              "inner join word on sense.wordid = word.wordid "
            "where word.lemma = '{}' "
        ).format(word)
        return self.conn.execute(query)

    def select_hyper(self, synset):
        query = (
            "select synset1 from synlink where link='hypo' and synset2 = '{}'"
        ).format(synset)
        return self.conn.execute(query)


class Wordnet():
    def __init__(self, config):
        self.config = config
        self.cache_collect_words = {}
        self.model = WordnetModel(config)

        self.hierarchy = self.create_hierarchy()
        self.reverse_hierarchy = self.create_reverse_hierarchy(self.hierarchy)

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

    def create_reverse_hierarchy(self, hierarchy):
        reverse = {}
        for parent, children in hierarchy.items():
            for synset in children:
                reverse[synset] = parent

        return reverse

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
            output |= self.__collect_synsets_recursive(name, hierarchy, 0)
            print(output)

        query = 'select name from synset where synset in ({})'.format(','.join(f':{i}' for i in range(len(output))))
        cur = self.config.conn.execute(query, tuple(output))
        print([row[0] for row in cur.fetchall()])

    def __collect_synsets_recursive(self, synset, hierarchy, depth):
        if 500 < depth:
            return set()

        if synset not in hierarchy.keys():
            return set()

        if synset in self.cache_collect_words:
            return self.cache_collect_words[synset]

        output = set()
        hypos = set(hierarchy[synset])
        output |= hypos

        depth += 1
        for hypo in hypos:
            output |= self.__collect_synsets_recursive(hypo, hierarchy, depth)

        self.cache_collect_words[synset] = output
        return output

    def count_hypo_synsets(self):
        hierarchy = self.create_hierarchy()
        for synset in hierarchy.keys():
            output = self.__collect_synsets_recursive(synset, hierarchy, 0)
            cur = self.config.conn.execute("select name from synset where synset = '{}'".format(synset))
            for row in cur:
                print('{}\t{}'.format(row[0], len(output)))

    def count_hypo_words(self, synsets=None):
        hierarchy = self.create_hierarchy()
        if synsets is None:
            synsets = hierarchy.keys()

        for target_synset in synsets:
            synsets = self.__collect_synsets_recursive(target_synset, hierarchy, 0)

            count = 0
            for synset in synsets:
                cur = self.config.conn.execute("select wordid from sense where synset = '{}'".format(synset))
                count += len(cur.fetchall())

            depth, root_synset_name = self.get_depth(target_synset, 0)

            cur = self.model.select_word_by_synset(target_synset)
            data = cur.fetchall()
            if len(data) != 0:
                row = data[0]
                if len(row) == 2:
                    print('{}\t{}\t{}\t{}({})'.format(row[0], row[1], count, root_synset_name, depth))
                else:
                    print('{}\tNone\t{}\t{}({})'.format(row[0], count, root_synset_name, depth))

    def get_depth(self, target_synset, depth):
        depth = 0
        synsets = [target_synset]
        synset = target_synset
        while synset in self.reverse_hierarchy:
            depth += 1
            synset = self.reverse_hierarchy[synset]
            if synset in synsets:
                self.config.logger.error('circular reference: target_synset {}, current synset: {}'.format(target_synset, synset))
                return 99999, synset
            synsets.append(synset)

        cur = self.model.select_word_by_synset(synset)
        name = synset
        for row in cur:
            name = row[0]
        self.config.logger.debug('END depth: {}, synset: {}'.format(depth, name))
        return depth, name

    def collect_hypo_words(self, word):
        hierarchy = self.create_hierarchy()
        output = self.__collect_synsets_recursive('08111783-n', hierarchy)
        cur = self.model.select_synset_by_word(word)
        data = cur.fetchall()
        if len(data) == 0:
            self.config.logger.info('{} is not found in wordnet'.format(word))
            sys.exit()

        for row in data:
            self.config.logger.debug('synset with word({}): {}'.format(word, row))
            root_synset = row[0]
            synsets = self.__collect_synsets_recursive(root_synset, hierarchy)
            self.config.logger.debug(synsets)
            for synset in synsets:
                cur = self.model.select_word_by_synset(synset)
                print(synset)
                for row in cur:
                    print('{}\t{}\t{}'.format(synset, row[0], row[1]))

    def collect_all_words(self):
        cur = self.config.conn.execute("select lemma from word")
        for row in cur:
            print(row[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('db', help='db file')
    parser.add_argument('--count_hypo_synsets', action='store_true', help='')
    parser.add_argument('--count_hypo_words', choices=['all', 'top_concept'], default=None, help='')
    parser.add_argument('--collect_synsets', default=None, help='')
    parser.add_argument('--collect_hypo_words', default=None, help='')
    parser.add_argument('--collect_all_words', action='store_true', help='')
    parser.add_argument('--loglevel', default='DEBUG')
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    args.conn = sqlite3.connect(args.db)

    wordnet = Wordnet(args)
    if args.count_hypo_synsets:
        wordnet.count_hypo_synsets()
        sys.exit()

    if args.count_hypo_words is not None:
        hierarchy = wordnet.create_hierarchy()
        synsets = None
        if args.count_hypo_words == 'top_concept':
            synsets = set(hierarchy.keys()) - set(sum(hierarchy.values(), []))

        wordnet.count_hypo_words(synsets)
        sys.exit()

    if args.collect_synsets is not None:
        wordnet.collect_synsets(args.collect_synsets)
        sys.exit()

    if args.collect_hypo_words is not None:
        wordnet.collect_hypo_words(args.collect_hypo_words)
        sys.exit()

    if args.collect_all_words:
        wordnet.collect_all_words()
        sys.exit()

