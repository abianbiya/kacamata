from tinydb import TinyDB, Query
from tinydb.operations import increment

id = 1

class DataStore:

    ekse = None
    kelas = None
    kacamata = None
    db = None

    def __init__(self):
        self.db = TinyDB('db/database.json')
        self.ekse = self.db.table('eksekusi')
        self.kelas = self.db.table('kelas')
        self.kacamata = Query()
        # self.initialize()

    def initialize(self):
        self.db.purge_table('kelas')
        self.kelas.insert_multiple([
            {
                "kelas": 0,
                "kacamata": "baca",
                "nama":"baca-00",
                "path": "static/img/class/0.png",
                "like" : 0,
                "predicted" : 0
            },{
                "kelas": 1,
                "kacamata": "baca",
                "nama":"baca-01",
                "path": "static/img/class/1.png",
                "like" : 0,
                "predicted" : 0
            },{
                "kelas": 2,
                "kacamata": "baca",
                "nama":"baca-02",
                "path": "static/img/class/2.png",
                "like" : 0,
                "predicted" : 0
            },{
                "kelas": 3,
                "kacamata": "baca",
                "nama":"baca-03",
                "path": "static/img/class/3.png",
                "like" : 0,
                "predicted" : 0
            },{
                "kelas": 4,
                "kacamata": "baca",
                "nama":"baca-04",
                "path": "static/img/class/4.png",
                "like" : 0,
                "predicted" : 0
            },{
                "kelas": 5,
                "kacamata": "baca",
                "nama":"baca-05",
                "path": "static/img/class/5.png",
                "like" : 0,
                "predicted" : 0
            },
            # -- baca-sun
            {
                "kelas": 0,
                "kacamata": "sung",
                "nama": "sung-00",
                "path": "static/img/clazz/0.png",
                "like": 0,
                "predicted": 0
            },{
                "kelas": 1,
                "kacamata": "sung",
                "nama":"sung-01",
                "path": "static/img/clazz/1.png",
                "like" : 0,
                "predicted" : 0
            },{
                "kelas": 2,
                "kacamata": "sung",
                "nama":"sung-02",
                "path": "static/img/clazz/2.png",
                "like" : 0,
                "predicted" : 0
            },{
                "kelas": 3,
                "kacamata": "sung",
                "nama":"sung-03",
                "path": "static/img/clazz/3.png",
                "like" : 0,
                "predicted" : 0
            },{
                "kelas": 4,
                "kacamata": "sung",
                "nama":"sung-04",
                "path": "static/img/clazz/4.png",
                "like" : 0,
                "predicted" : 0
            },{
                "kelas": 5,
                "kacamata": "sung",
                "nama":"sung-05",
                "path": "static/img/clazz/5.png",
                "like": 0,
                "predicted": 0
            },{
                "kelas": 6,
                "kacamata": "sung",
                "nama":"sung-06",
                "path": "static/img/clazz/6.png",
                "like": 0,
                "predicted" : 0
            }]

        )

    def arsip(self, index, kelas, kelamin, kacamata, like):
        self.ekse.insert({
            'indeks_morfo' : index,
            'kelas': kelas,
            'jenis_kelamin': kelamin,
            'kacamata': kacamata,
            'like': like
        })
        return True, 'Berhasil insert data!'

    def like(self, kacamata, is_like):
        # print(kacamata)
        Kacamata = Query()
        ini = self.kelas.search((Kacamata.nama == kacamata))[0]
        # print(ini.doc_id)

        if(is_like == True):
            self.kelas.update({'like': ini['like']+1}, doc_ids=[ini.doc_id])

        return True


    def all_arsip(self):
        return self.ekse.all()

    def clear_arsip(self):
        self.db.purge_table('eksekusi')
        return True

    def all_kacamata(self, split=False):
        if split:
            Kacamata = Query()
            return self.kelas.search(Kacamata.kacamata == 'baca'), self.kelas.search(Kacamata.kacamata == 'sung'), 
        else:
            return self.kelas.all()


# data = DataStore()
# for i in data.all_kacamata():
#     print(i)

