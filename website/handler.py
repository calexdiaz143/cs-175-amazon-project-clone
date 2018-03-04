from django.shortcuts import render
from django.http import HttpResponse
import urllib.parse, json
import main.main, main.parser, main.trainer

def index(request):
    if request.method == 'POST':
        uri_review = request.POST.get('review', '{}')
        json_review = urllib.parse.unquote(uri_review)
        review = json.loads(json_review)

        if review['helpful'][0] == None or review['helpful'][1] == None:
            review['helpful'] = [0, 0]
        review = main.parser.parse_review(review)

        classifier = main.trainer.load('/app/main/static/clf_log')
        prediction = main.main.predict(review, classifier, '/app/main/static/summary_cv.pkl', '/app/main/static/review_cv.pkl')

        return HttpResponse(prediction)
    else:
        return render(request, 'index.html')

def predict(review, classifier):
    import parser
    import numpy as np
    from scipy.sparse import csr_matrix, hstack

    review = csr_matrix(review)
    tr, te = parser.parse_BOW([], review)

    classifier_naive_bayes.predict(csr_matrix(review))
