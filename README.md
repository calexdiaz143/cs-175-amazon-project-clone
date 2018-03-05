# cs175-amazon-project
A text-based machine learning thingy.

## Database
[Amazon Review Data](http://jmcauley.ucsd.edu/data/amazon/)

## Docs
[Project Proposal](https://docs.google.com/document/d/1hshj-fZLoi63BUrHVJ_Q99C_sCyEwYaukFC7FMrQfD0/edit)  
[Project Progress Report](https://docs.google.com/document/d/1Wwyn0p2aMKDBf04hzSTeHaJUC_Zz39L2VguePel3p94/edit)  
[Project Presentation](https://docs.google.com/presentation/d/1fKKkVUE7hq4tzrj18FGuS0dxemu1tBUQI1z_azUvke8/edit)

## Apps
[Nonfunctional Website](https://amazonpredictor.appspot.com/)  
[Functional Website](https://amazonpredictor.herokuapp.com/)

## TODO
- implement the tf-idf vectorizer
- something about n-grams
- change our method of parsing helpfulness
  - current:
    - set helpful=[0,0] to 0.5
    - set helpful=[x,y] to x/y
    - throw out anything under 0.5
  - proposed:
    - set helpful=[x,y] to log(x-(y-x))
      - (i.e. logarithm of helpful minus unhelpful)
    - keep everything, and keep helpfulness as a feature
- implement a hash vectorizer maybe
- filter reviews by language
- fix reviewerID (error b/c it's too large)
- include reviewerName as a feature
  - b/c e.g. there's a guy named (Mike Tarrani "Jazz Drummer")
