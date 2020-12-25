import os
import tweepy
import ruamel.yaml
import json
import argparse
from datetime import date, datetime
from logzero import setup_logger

class Collector():
    def __init__(self, config):
        self.config = config

        with open('key.yaml', 'r') as f:
            yaml = ruamel.yaml.round_trip_load(f)

        auth = tweepy.OAuthHandler(yaml['consumer_key'], yaml['consumer_secret'])
        auth.set_access_token(yaml['access_token'], yaml['access_token_secret'])
        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def collect_tweets_with_hashtags(self, input):
        self.config.logger.info(f'input: {input}')        
        hashtags = [input]
        if os.path.isfile(input):
            with open(input) as f:
                hashtags = [line.strip() for line in f.readlines()]

        for word in hashtags:
            self.collect_tweets_with_hashtag(word)

    def collect_tweets_with_hashtag(self, hashtag):
        self.config.logger.info(f'hashtag: {hashtag}')        

        output_path = self.config.output_path
        if self.config.output_path is None:
            output_path = f'{hashtag}.json'

        status_list = tweepy.Cursor(self.api.search
                                    , q=f'#{hashtag} -filter:retweets'
                                    , lang='ja'
                                    , tweet_mode = 'extended'
                                    ).items(self.config.count)
        for status in status_list:
            result = {}
            result['id'] = status.id
            result['created_at'] = status.created_at
            result['name'] = status.user.screen_name
            result['text'] = status.full_text
            with open(output_path, 'a') as f:
                json.dump(result, f, default=json_serial, indent=2, ensure_ascii=False)

        self.config.logger.info(f'done: {hashtag}')        

def json_serial(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    raise TypeError ("Type %s not serializable" % type(obj))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('hashtag')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--loglevel', default='DEBUG')
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    collector = Collector(args)
    collector.collect_tweets_with_hashtags(args.hashtag)
