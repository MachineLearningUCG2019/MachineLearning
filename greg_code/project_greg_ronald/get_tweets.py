import json, tweepy, re

creds = json.load(open("creds.json"))

auth = tweepy.OAuthHandler(creds["CONSUMER_KEY"], creds["CONSUMER_SECRET"])
auth.set_access_token(creds["ACCESS_KEY"], creds["ACCESS_SECRET"])

api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
tweets = []

r = re.compile(' (https|http):\/\/t\.co\/[a-zA-Z0-9]{0,10}')
def remove_links(tweet):
  return r.sub("", tweet)

def save_state(uname):
  print("saving...")
  f = open("original_tweets/"+uname+".txt","w",encoding="utf-8")
  f.write("\n".join(tweets))
  f.close()
  f = open("original_tweets/"+uname+"_linkless.txt","w",encoding="utf-8")
  f.write("\n".join(list(map(remove_links,tweets))))
  f.close()
  print("saved")
  return True

def main(username):
  i=0
  print("getting info")
  user_info = api.get_user(username)._json
  f = open("original_tweets/info/"+username+".json","w")
  f.write(json.dumps(user_info))
  f.close()
  print("info saved")
  for tweet in tweepy.Cursor(api.user_timeline,id=username,include_rts=False).items():
    tweets.append(tweet.text)
    print(i,tweet.text.encode("utf-8"))
    i += 1
    if(not (i % 100)):
      save_state(username)
  save_state(username)
  print("count",i)
  print(len(tweets))
  return True

main(input("who are we stalking?: "))