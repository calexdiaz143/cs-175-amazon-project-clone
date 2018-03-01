import webapp2
import json
import pickle
# import predictor

class Index(webapp2.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/html'
        content = open('index.html').read()
        self.response.write(content)

class Predictor(webapp2.RequestHandler):
    def post(self):
        self.response.headers['Content-Type'] = 'application/json; charset=UTF-8'
        json_review = self.request.get('review', 'undefined')
        raw_review = json.loads(json_review)

        review = [
            1, 1, 1, # TODO: fix this after it's fixed in loader
        	# int(raw_review['reviewerID'], 36),
        	# int(raw_review['asin'], 36),
        	# loader.get_helpful_percentage(raw_review['helpful'],
        	int(raw_review['overall']),
        	raw_review['unixReviewTime'],
        	raw_review['summary'],
        	raw_review['reviewText']
        ]

        # clf = pickle.load(open('root_copy/saved/clf.pkl', 'rb'))

        # train_X, train_Y, test_X, test_Y = loader.load([], True)
        # classifier = trainer.naive_bayes(train_X, train_Y)
        # prediction = classifier.predict([review])

        self.response.write(json.dumps(review))

sitemap = [
    ('/', Index),
    ('/predict', Predictor)
]

app = webapp2.WSGIApplication(sitemap, debug=True)
