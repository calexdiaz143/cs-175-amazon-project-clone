from django.shortcuts import render
from django.http import HttpResponse
import urllib.parse, json
import main.memo, main.loader, main.parser

def index(request):
    if request.method == 'POST':
        uri_review = request.POST.get('review', '{}')
        json_review = urllib.parse.unquote(uri_review)

        review = json.loads(json_review)
        if review['helpful'][0] == None or review['helpful'][1] == None:
            review['helpful'] = [0, 0]
        raw_review = main.loader.parse_review(review)

        summary_cv = main.memo.load_pkl('/app/main/static/summary_cv')
        review_cv = main.memo.load_pkl('/app/main/static/review_cv')
        review = main.parser.transform([raw_review], summary_cv, review_cv)

        clf_nb = main.memo.load_pkl('/app/main/static/clf_lr')
        clf_bnb = main.memo.load_pkl('/app/main/static/clf_lr')
        clf_lr = main.memo.load_pkl('/app/main/static/clf_lr')
        clf_rf = main.memo.load_pkl('/app/main/static/clf_lr')
        clf_gb = main.memo.load_pkl('/app/main/static/clf_lr')
        predictions = [
            clf_nb.predict(review),
            clf_bnb.predict(review),
            clf_lr.predict(review),
            clf_rf.predict(review),
            clf_gb.predict(review)
        ]
        prediction = main.tester.predict_ensemble(review, predictions)

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
