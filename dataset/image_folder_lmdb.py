# adapted from: https://github.com/rmccorm4/PyTorch-LMDB/blob/master/folder2lmdb.py

import lmdb
import os
import six

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
# This segfaults when imported before torch: https://github.com/apache/arrow/issues/2637
import pyarrow as pa
torch.multiprocessing.set_start_method('spawn')

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

if __name__ == "__main__":
    from imagenet_names import names

    db_path = "D:\\FYPCode\\ILSVRC2012\\val.lmdb"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    dataset = ImageFolderLMDB(db_path, transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    for item in trainloader:
        print(item)
        break

    # for i in range(5):
    #     sample_img, sample_target = dataset[i]
    #     sample_img.show()
    #     print('sample target at i: ', names[sample_target])
    #     input("Press Enter to continue...")