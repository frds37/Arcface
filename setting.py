import easydict

setting = easydict.EasyDict()

setting.batch_size = 256
setting.lr = 0.1
setting.momentum = 0.9
setting.weight_decay = 5e-4
setting.epoch_num = 34
setting.embedding_size = 512
setting.num_class = 10572
setting.folder = '/home/cryptology96/FaceRecognition/Challenge1/DS'
setting.target = ['lfw', 'cfp_fp', 'agedb_30']