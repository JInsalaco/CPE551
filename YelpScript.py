# coding: utf8
import requests
import csv
from bs4 import BeautifulSoup

#Define the page url for Yelp. Yelp uses the following url scheme for reviews https://www.yelp.com/biz/kenka-new-york?start={numStart}
#where numStart is a page number in multiples of 10
numStart = 10
reviews=[]

#Using beautifulsoup, we can grab the html of every page for our restaurant
for x in range(0,10):
    #builds a dynamic url using f string 
    url = f'https://www.yelp.com/biz/kenka-new-york?start={numStart}'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    #parse out the text reviews and star ratings
    text = [textReview.text.strip() for textReview in soup.findAll('span', {'class' : 'raw__373c0__3rcx7'})]
    rate = [nr for nr in soup.findAll('div', {'class':'i-stars__373c0__1T6rz'})]
    
    #yelp gives us a bunch of junk within the page such as ratings and reviews from other restaurants.
    #All reviews for our chosen url are contained within these ranges
    text = text[4:14]
    rate = rate[1:11]

    #pull out the aria-label value from the html
    for i in range(0,len(rate)):
        rate[i] = rate[i]['aria-label']
    
    #create the tuple containing the review and rating we found
    for i in range(0, len(rate)):
        reviews.append([text[i], rate[i]])
    numStart+=10

#write the csv file so we do not need to continuously pull data from yelp
#If we call this program too many times, you may get timed out/banned from yelp (I did)
headers = ['Review', 'Rating']
filename = "kanka_yelp_reviews.csv"
with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(headers)
    csvwriter.writerows(reviews)