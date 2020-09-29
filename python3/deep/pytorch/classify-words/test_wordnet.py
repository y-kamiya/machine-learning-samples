# -*- coding: utf-8 -*-
import sys
import unittest
import sqlite3

from wordnet import Wordnet

class TestWordnet(unittest.TestCase):
    DB_PATH = 'data/wnjpn.db'

    class DummyArgs(object):
        pass

    class DummyLogger(object):
        def debug(self, _): pass
        def info(self, _): pass
        def warning(self, _): pass
        def error(self, _): pass

    def setUp(self):
        args = self.DummyArgs()
        args.conn = sqlite3.connect(self.DB_PATH)
        args.logger = self.DummyLogger()
        self.instance = Wordnet(args)

    def test_get_depth(self):
        synset_entity = '00001740-n'
        depth, name = self.instance.get_depth(synset_entity, 0)
        self.assertEqual(depth, 0)
        self.assertEqual(name, 'entity')

        synset_attribute = '00024264-n'
        depth, name = self.instance.get_depth(synset_attribute, 0)
        self.assertEqual(depth, 2)
        self.assertEqual(name, 'entity')


if __name__ == '__main__':
    unittest.main(verbosity=2)

