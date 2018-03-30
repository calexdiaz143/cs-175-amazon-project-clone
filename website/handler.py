from django.shortcuts import render
from django.http import HttpResponse
import urllib.parse, json
import main.memo, main.loader, main.parser, main.tester

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

        clf_nb = main.memo.load_pkl('/app/main/static/clf_nb')
        clf_bnb = main.memo.load_pkl('/app/main/static/clf_bnb')
        # clf_rf = main.memo.load_pkl('/app/main/static/clf_rf') # excluded due to large file size (>7GB)
        # clf_gb = main.memo.load_pkl('/app/main/static/clf_gb') # excluded due to long prediction time
        clf_lr = main.memo.load_pkl('/app/main/static/clf_lr')
        predictions = [
            clf_nb.predict(review),
            clf_bnb.predict(review),
            # clf_rf.predict(review),
            # clf_gb.predict(review),
            clf_lr.predict(review)
        ]
        prediction = main.tester.predict_ensemble(review, predictions)

        return HttpResponse(prediction)
    else:
        return render(request, 'index.html')

def predict(review, classifier):
    import main.parser
    import numpy as np
    from scipy.sparse import csr_matrix, hstack

    review = csr_matrix(review)
    tr, te = main.parser.parse_BOW([], review)

    classifier_naive_bayes.predict(csr_matrix(review))
