
from bs4 import BeautifulSoup
from urllib.parse import urlparse

import praw
import time
import re
import requests
import bs4
import numpy as np


def authenticate():
    


	print('Authenticating...\n')


	reddit = praw.Reddit('mysterybot123', user_agent = 'mystery bot for stupid stuff')
	print('Authenticated as {}\n'.format(reddit.user.me()))

	return reddit



def run_explainbot(reddit):
    
	print("Getting 250 comments...\n")
	
	file = open("comments.txt", "w")

	for comment in reddit.subreddit('wallstreetbets').comments(limit = 550):
		cm_body  = comment.body
		file.write(cm_body)

	file.close()

def get_tickers(reddit):
	stock = {}

	for comment in reddit.subreddit('wallstreetbets').comments(limit = 550):
		match = re.findall("[A-Z0-9a-z ]+\$(.*)", comment.body)
		if match:

			#print(comment.body)
			ticker = comment.body.split('$')[1].split(' ')[0];
			mat = re.findall("[A-Z]+", ticker)
			if mat:
				stock[mat[0]] = comment 
		

	for tick in stock:
		print(tick)		
	print('Waiting 60 seconds...\n')


myreddit = authenticate()

run_explainbot(myreddit)


