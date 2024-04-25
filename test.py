import numpy as np
import os
from utils import metric as M
from PIL import Image


class test_dataloader:
    def __init__(self, salmap_root, gt_root):
        self.salmap = [salmap_root + f for f in os.listdir(salmap_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.salmap = sorted(self.salmap)
        # self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self.size = len(self.salmap)
        self.index = 0

    def load_data(self):
        salmap = self.binary_loader(self.salmap[self.index])
        gt = self.binary_loader(self.gts[self.index])
        name = self.salmap[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return salmap, gt, name

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


dataset_path = 'D:/MODULE/dataset/'
test_datasets = ['EORSSD', 'ORSSD']
salmap_root = 'D:/MODULE/result/Maps/Final_DHCont_MCCNet/'  # 'D:/MODULE/result/Maps/Final_DHCont_ACCoNet/'
for dataset in test_datasets:
    save_path = './results/VGG/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(dataset)
    gt_root = dataset_path + dataset + '/test-labels/'
    test_loader = test_dataloader(salmap_root + dataset + '/', gt_root)

    FM = M.Fmeasure_and_FNR()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()
    for i in range(test_loader.size):
        salmap, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        pred = np.asarray(salmap, np.float32)
        pred = pred / 255
        FM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)

    fm = FM.get_results()[0]['fm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']

    adpFm = fm['adp'].round(4)
    meanFm = fm['curve'].mean().round(4)
    maxFm = fm['curve'].max().round(4)
    maxEm = em['curve'].max().round(4)
    meanEm = em['curve'].mean().round(4)
    adpEm = em['adp'].round(4)
    smeasure = sm.round(4)
    mae_mean = mae.round(6)
    print('maxFm {:.4f}, meanFm {:.4f}, adpFm {:.4f}, maxEm {:.4f}, meanEm {:.4f}, adpEm {:.4f},'
          ' smeasure {:.4f}, mae_mean {:.6f}'.format(maxFm, meanFm, adpFm, maxEm, meanEm, adpEm, smeasure, mae_mean))
