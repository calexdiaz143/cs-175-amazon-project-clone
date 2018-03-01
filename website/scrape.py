import websoc
import antndb

def scrape(year=None, term=None):
    if not year:
        year = websoc.fetch.YEARS[-1]
        keys = None
    else:
        keys = antndb.get(keys_only=True)
    data = websoc.fetch.iterwebsoc(year, term)

    database = {}
    for document in data:
        websoc.parse.parsedocument(database, document)

    for building in database:
        b = antndb.Building(
            name=building,
        )

        for room in database[building]:
            entity = antndb.Room(
                id=' '.join([building, room]),
                sunday=database[building][room]['Su'].list,
                monday=database[building][room]['M'].list,
                tuesday=database[building][room]['Tu'].list,
                wednesday=database[building][room]['W'].list,
                thursday=database[building][room]['Th'].list,
                friday=database[building][room]['F'].list,
                saturday=database[building][room]['Sa'].list,
                initial_yearterm=''.format(year),
                final_yearterm=''.format(year)
            )
            key = entity.put()
            if keys:
                if key in keys:
                    keys.remove(key)

    if keys:
        entities = antndb.ndb.get_multi(keys)
        for entity in entities:
            entity.su = entity.mo = entity.tu = entity.we = entity.th = entity.fr = entity.sa = '[]'
        antndb.ndb.put_multi(entities)

if __name__ == '__main__':
    scrape()
