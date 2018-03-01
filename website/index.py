import webapp2

class Index(webapp2.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/html'
        content = open('index.html').read()
        self.response.write(content)

class PostLocation(webapp2.RequestHandler):
    def post(self):
        data = self.request.get('data', None)
        jsondata = json.loads(data)
        Location(data=jsondata).put()
        self.response.write('Success. Your location has been added to the database.')

class GetLocation(webapp2.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/javascript'
        location = Location.query().fetch()
        content = [l.data for l in location]
        self.response.write(content)

sitemap = [
    ('/', Index),
    ('/post', PostLocation),
    ('/get', GetLocation)
]

app = webapp2.WSGIApplication(sitemap, debug=True)
